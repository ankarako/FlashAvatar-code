###################################################################
# Finds gaussian avatars - nersemble camera translation based on 
# 2d lanmark alignment.
###################################################################
import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
import argparse
from scene import SceneGaussianAvatars
from easydict import EasyDict
from ddfav3.model.recon import face_model
from ddfav3.face_box import face_box
import cv2
import t3d
from scene.cameras import CameraGaussianAvatars
from PIL import Image
from ddfav3.fit_video import process_lmks

k_iscrop = True
k_detector = "retinaface"
k_ldm68 = True
k_ldm106_2d = True
k_ldm134 = True
k_seg = True
k_seg_visible = True
k_use_tex = True
k_extract_tex = True
k_backbone = "resnet50"
# default device is gpu
k_device = "cuda"

k_face_model_args = EasyDict(
    device=k_device,
    ldm68=k_ldm68,
    ldm106=k_ldm106_2d,
    ldm106_2d=k_ldm106_2d,
    ldm134=k_ldm134,
    seg_visible=k_seg_visible,
    seg=k_seg,
    backbone=k_backbone,
    extractTex=k_extract_tex
)

k_facebox_args = EasyDict(
    iscrop=k_iscrop,
    detector=k_detector,
    device=k_device
)

k_class_labels = {
    'leye': 1,
    'reye': 2,
    'leyebrow': 3,
    'reyebrow': 4,
    'nose': 5,
    'ulip': 6,
    'blip': 7,
    'face': 8
}

k_class_colors = {
    0: np.array([0, 0, 255]),
    1: np.array([0, 255, 0]),
    2: np.array([0, 255, 255]),
    3: np.array([255, 0, 0]),
    4: np.array([255, 0, 255]),
    5: np.array([255, 255, 0]),
    6: np.array([255, 255, 0]),
    7: np.array([255, 255, 255]),
}

k_colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

k_flame_kwargs = {
    'model_path': 'flame/flame2023/flame2023.pkl',
    'mesh_path': 'flame/flame2023/head_template_mesh.obj',
    'masks_path': 'flame/flame2023/FLAME_masks.pkl',
    'lmk_embedding_with_eyes_path': 'flame/flame2023/landmark_embedding_with_eyes.npy',
    'nid_params': 300,
    'nex_params': 100,
    'add_teeth': True
}

def draw_lmks_2d(img: np.ndarray, lmks: np.ndarray, color: str) -> torch.Tensor:
    """
    Draw the specified landmarks on the given image

    :param img The image on which to draw the landmarks
    :param lmks The landmarks to draw
    :param color The color to use
    """
    # convert to numpy
    for lmk in lmks:
        img = cv2.circle(img, (round(lmk[0].item()), round(lmk[1].item())), 5, color=k_colors[color], thickness=-1)
    # convert to torch again
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga_data_root", type=str, help="The root directory of the GaussianAvatars-Nersemble scene")
    args = parser.parse_args()

    # create scene
    scene = SceneGaussianAvatars(args.ga_data_root, "transforms_train.json")

    # create face model
    recon_model = face_model(k_face_model_args)
    facebox_detector = face_box(k_facebox_args).detector

    # load flame
    flame_model = t3d.Flame2023(**k_flame_kwargs).to(k_device)

    # gather all the different cameras in the dataset
    cameras = []
    for camera in scene.cameras:
        if camera.colmap_id < 15:
            cameras += [camera]
    
    # for each camera
    camera: CameraGaussianAvatars
    for camera in cameras:
        fx = t3d.cam.param.fov2focal(camera.FoVx, camera.image_width)
        fy = t3d.cam.param.fov2focal(camera.FoVy, camera.image_height)
        # load the associate camera image
        rgb = Image.open(camera.image_name).convert('RGB')
    
        # detect lmks
        trans_params, im_tensor = facebox_detector(rgb)
        recon_model.input_img = im_tensor.to(k_device)
        results = recon_model.forward()
        lmks68 = results['ldm68'][0]
        lmks68, img_lmks68 = process_lmks(lmks68, trans_params, rgb)
        
        lmks68 = torch.from_numpy(lmks68).to(k_device)
        t_opt = torch.zeros([1, 3], device=k_device, requires_grad=True)

        optim = torch.optim.Adam([t_opt], lr=1.0e-4)
        for iter in range(5000):    
            K = torch.from_numpy(camera.K).to(dtype=torch.float32).to(k_device)
            intrin = torch.eye(4, device=k_device)
            intrin[:3, :3] = K
            intrin = intrin.unsqueeze(0)

            R = torch.from_numpy(camera.R).to(dtype=torch.float32, device=k_device).unsqueeze(0)
            T = torch.from_numpy(camera.T).to(dtype=torch.float32, device=k_device).unsqueeze(0) + t_opt
            
            cam_p3d = PerspectiveCameras(
                K=intrin,
                R=R,
                T=T,
                image_size=(camera.image_height, camera.image_width)
            ).to(k_device)

            trn = cam_p3d.get_world_to_view_transform()

            verts, lmks = flame_model.forward(
                scene.shape_param.to(k_device),
                camera.exp_param.to(k_device),
                camera.rot.to(k_device),
                camera.neck_pose.to(k_device),
                camera.jaw_pose.to(k_device),
                camera.eyes_pose.to(k_device),
                camera.flame_trans.to(k_device),
                return_lmks=True
            )

            flame_lmks2d = cam_p3d.transform_points_screen(lmks)

            # perform an l1 loss on the lmks
            loss = torch.abs(flame_lmks2d[0, :68, 0:2] - lmks68).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # show lmks image
            output = np.ones([camera.image_height, camera.image_width, 3]) * 255
            output_lmks68 = draw_lmks_2d(output, lmks68.detach().cpu().numpy(), 'green')
            output_all = draw_lmks_2d(output_lmks68, flame_lmks2d.detach().cpu().numpy()[0, :68, 0:2], 'red')

            cv2.imshow('optim', output_all)
            cv2.waitKey(1)