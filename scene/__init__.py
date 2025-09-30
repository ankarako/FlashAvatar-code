from typing import List
import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera, CameraGaussianAvatars
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov, fov2focal

import tloaders


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 0
            range_up = train_num
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)
            
            flame_params = payload['flame']
            exp_param = torch.as_tensor(flame_params['exp'])
            eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(images_folder, image_name_ori+'.jpg')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras





    
class SceneGaussianAvatars:
    """
    Implements Nersemble gausian avatars scene dataloading.
    """
    def __init__(
        self,
        data_root: str,
        transforms_filename: str,
        prefetch: bool=False
    ) -> None:
        """
        Initialize a SceneGaussianAvatars object

        :param data_root
        :param transforms_filename
        :param prefetch
        """
        self.dataset: tloaders.GaussianAvatars = tloaders.DatasetRegistry.get_dataset(
            "GaussianAvatars",
            data_root=data_root,
            transforms_filename=transforms_filename,
            prefetch=prefetch
        )

        # parse our flame samples to Camera objects
        self.cameras = []
        parsing_loop = tqdm(
            enumerate(self.dataset.cam_infos),
            total=len(self.dataset.cam_infos),
            desc="Parsing GaussianAvatars to Scene"
        )
        sample: tloaders.NersembleCamInfo
        for idx, sample in parsing_loop:
            K = sample.K
            fx = K[0, 0]
            fy = K[1, 1]
            fovx = sample.fovx
            fovy = focal2fov(fov2focal(fovx, sample.width), sample.height)

            
            c2w = sample.c2w
            width, height = sample.width, sample.height
            shape_params = sample.flame_id_params
            expr_params = sample.flame_ex_params
            fl_rot = sample.flame_rotation
            fl_neck = sample.flame_neck
            fl_jaw = sample.flame_jaw
            fl_eyes = sample.flame_eyes
            fl_trans = sample.flame_translation
            fl_static_offsets = sample.flame_static_offset

            image_path = sample.image_path
            msk_path = sample.mask_name
            
            self.shape_param = torch.from_numpy(shape_params).unsqueeze(0)
            self.static_offset = torch.from_numpy(fl_static_offsets)

            # dataset must be proprocessed with
            # face_parsing.
            # see preproc_ga_fparsing.py
            mask_dir = os.path.dirname(msk_path)
            mask_fname = os.path.basename(msk_path)
            mouth_mask_path = os.path.join(mask_dir, '..', 'parsing', f'{mask_fname.replace(".png","")}_mouth.png')
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            # T[1] -= 0.065
            # T = sample.campos
            

            # fl_trans[:, 1] += 0.07
        

            self.cameras += [CameraGaussianAvatars(
                uid = sample.uid,
                R=R,
                T=T,
                FoVx=fovx,
                FoVy=fovy,
                image_path=image_path,
                mask_path=msk_path,
                mouth_mask_path=mouth_mask_path,
                image=None,
                head_mask=None,
                mouth_mask=None,
                exp_param=expr_params,
                eyes_pose=fl_eyes,
                eyelids=None,
                jaw_pose=fl_jaw,
                neck_pose=fl_neck,
                rot=fl_rot,
                fl_trans=fl_trans,
                image_name=image_path,
                width=width,
                height=height,
                colmap_id=sample.uid,
                trans=np.array([0.0, -0.078, 0.0]),
                campos=torch.from_numpy(sample.campos).to(dtype=torch.float32)
            )]

    
    def getCameras(self) -> List[Camera]:
        return self.cameras

