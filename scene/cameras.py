#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(
        self, 
        colmap_id, 
        R, 
        T, 
        FoVx, 
        FoVy, 
        image_path,
        mask_path,
        mouth_mask_path,
        image, 
        head_mask, 
        mouth_mask,
        exp_param, 
        eyes_pose, 
        eyelids, 
        jaw_pose,
        image_name, 
        width,
        height,
        uid,
        trans=np.array([0.0, 0.0, 0.0]), 
        scale=1.0, 
        data_device = "cpu"
    ) -> None:
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_path = image_path
        self.original_image = None
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

        self.image_width = width
        self.image_height = height

        self.head_mask_path = mask_path
        self.head_mask = None
        if head_mask is not None:
            self.head_mask = head_mask.to(self.data_device)
        
        self.mouth_mask_path = mouth_mask_path
        self.mouth_mask = None
        if mouth_mask is not None:
            self.mouth_mask = mouth_mask.to(self.data_device)
        
        exp_param = torch.from_numpy(exp_param)
        self.exp_param = exp_param.to(self.data_device)

        eyes_pose = torch.from_numpy(eyes_pose)
        self.eyes_pose = eyes_pose.to(self.data_device)

        if eyelids is not None:
            self.eyelids = eyelids.to(self.data_device)
        
        jaw_pose = torch.from_numpy(jaw_pose)
        self.jaw_pose = jaw_pose.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]



class CameraGaussianAvatars(torch.nn.Module):
    def __init__(
        self, 
        colmap_id, 
        R, 
        T, 
        FoVx, 
        FoVy, 
        image_path,
        mask_path,
        mouth_mask_path,
        image, 
        head_mask, 
        mouth_mask,
        exp_param, 
        eyes_pose, 
        eyelids, 
        jaw_pose,
        neck_pose,
        rot,
        fl_trans,
        image_name, 
        width,
        height,
        uid,
        campos,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
    ) -> None:
        super(CameraGaussianAvatars, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_path = image_path
        self.original_image = None
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

        self.image_width = width
        self.image_height = height

        self.head_mask_path = mask_path
        self.head_mask = None
        if head_mask is not None:
            self.head_mask = head_mask.to(self.data_device)
        
        self.mouth_mask_path = mouth_mask_path
        self.mouth_mask = None
        if mouth_mask is not None:
            self.mouth_mask = mouth_mask.to(self.data_device)
        
        exp_param = torch.from_numpy(exp_param)
        self.exp_param = exp_param.to(self.data_device)

        eyes_pose = torch.from_numpy(eyes_pose)
        self.eyes_pose = eyes_pose.to(self.data_device)

        if eyelids is not None:
            self.eyelids = eyelids.to(self.data_device)
        
        jaw_pose = torch.from_numpy(jaw_pose)
        self.jaw_pose = jaw_pose.to(self.data_device)

        neck_pose = torch.from_numpy(neck_pose)
        self.neck_pose = neck_pose.to(self.data_device)

        rot = torch.from_numpy(rot)
        self.rot = rot.to(self.data_device)

        flame_trans = torch.from_numpy(fl_trans)
        self.flame_trans = flame_trans.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_center = campos.to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

