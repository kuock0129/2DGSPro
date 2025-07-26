#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# 2DGSPro

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
import torch.nn.functional as F
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel2D:

    def setup_functions(self):
        # Build the covariance matrix from scaling, rotation and translation
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        # Activation functions for the model parameters
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # Build the covariance matrix from scaling, rotation and translation
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self._tangent_u = torch.empty(0)  # First tangent vector [N, 3]
        self._tangent_v = torch.empty(0)  # Second tangent vector [N, 3]

        self.setup_functions()
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._tangent_u,  # Add tangent vectors to capture
            self._tangent_v,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 14:  # New format with tangent vectors
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._tangent_u,
            self._tangent_v) = model_args
        else:  # Old format without tangent vectors
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            # Initialize tangent vectors for backward compatibility
            N = self._xyz.shape[0]
            self._tangent_u, self._tangent_v = self.create_random_orthogonal_tangents_batch(N)
            
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_tangent_u(self):
        if self._tangent_u.numel() == 0:
            return torch.empty(0, 3, device="cuda")
        return F.normalize(self._tangent_u, dim=-1)
    
    @property 
    def get_tangent_v(self):
        if self._tangent_v.numel() == 0:
            return torch.empty(0, 3, device="cuda")
        # Ensure orthogonality through Gram-Schmidt process
        u = self.get_tangent_u
        v_raw = self._tangent_v
        
        # Remove component parallel to u
        v_orthogonal = v_raw - torch.sum(v_raw * u, dim=-1, keepdim=True) * u
        
        return F.normalize(v_orthogonal, dim=-1)

    @property
    def get_normals(self):
        """Use tangent-based normal computation as intended"""
        if self._tangent_u.numel() == 0 or self._tangent_v.numel() == 0:
            # Fallback: quaternion-based normals if tangent vectors not available
            return self.get_normals_from_rotation()
            
        # Method 1: From tangent vectors 
        tangent_u = F.normalize(self._tangent_u, dim=-1)
        tangent_v_raw = self._tangent_v
        
        # Ensure orthogonality through Gram-Schmidt
        tangent_v = tangent_v_raw - torch.sum(tangent_v_raw * tangent_u, dim=-1, keepdim=True) * tangent_u
        tangent_v = F.normalize(tangent_v, dim=-1)
        
        # Normal = cross product of tangents
        normals = torch.cross(tangent_u, tangent_v, dim=-1)
        return F.normalize(normals, dim=-1)

    def get_normals_from_rotation(self):
        """Fallback normal computation from rotation quaternions"""
        if self._rotation.numel() == 0:
            return torch.empty(0, 3, device="cuda")
            
        rotations = self.get_rotation()  # [N, 4] quaternions
        w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        
        # Rotation matrix from quaternion
        rot_matrices = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=1)
        ], dim=1)  # [N, 3, 3]
        
        normals = rot_matrices[:, :, 2]  # [N, 3]
        return F.normalize(normals, dim=-1)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # initialize Gaussians (scales, rotations, opacities) from a raw point cloud
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        # TANGENT VECTOR INITIALIZATION
        N = fused_point_cloud.shape[0]
        tangent_u, tangent_v = self.create_random_orthogonal_tangents_batch(N)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._tangent_u = nn.Parameter(tangent_u.requires_grad_(True))  # FIX: Add tangent vectors as parameters
        self._tangent_v = nn.Parameter(tangent_v.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},  # FIX: Added missing comma
        ]
        
        # Add tangent vectors to optimizer if they exist
        if hasattr(self, '_tangent_u') and self._tangent_u.numel() > 0:
            l.extend([
                {'params': [self._tangent_u], 'lr': training_args.rotation_lr, "name": "tangent_u"},
                {'params': [self._tangent_v], 'lr': training_args.rotation_lr, "name": "tangent_v"}
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add tangent vector attributes
        if hasattr(self, '_tangent_u') and self._tangent_u.numel() > 0:
            l.extend(['tangent_u_x', 'tangent_u_y', 'tangent_u_z'])
            l.extend(['tangent_v_x', 'tangent_v_y', 'tangent_v_z'])
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        
        # Add tangent vectors if they exist
        if hasattr(self, '_tangent_u') and self._tangent_u.numel() > 0:
            tangent_u = self._tangent_u.detach().cpu().numpy()
            tangent_v = self._tangent_v.detach().cpu().numpy()
            attributes.extend([tangent_u, tangent_v])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes_concat = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes_concat))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load tangent vectors if they exist in the PLY file
        property_names = [p.name for p in plydata.elements[0].properties]
        if 'tangent_u_x' in property_names:
            tangent_u = np.stack([
                np.asarray(plydata.elements[0]["tangent_u_x"]),
                np.asarray(plydata.elements[0]["tangent_u_y"]),
                np.asarray(plydata.elements[0]["tangent_u_z"])
            ], axis=1)
            tangent_v = np.stack([
                np.asarray(plydata.elements[0]["tangent_v_x"]),
                np.asarray(plydata.elements[0]["tangent_v_y"]),
                np.asarray(plydata.elements[0]["tangent_v_z"])
            ], axis=1)
            self._tangent_u = nn.Parameter(torch.tensor(tangent_u, dtype=torch.float, device="cuda").requires_grad_(True))
            self._tangent_v = nn.Parameter(torch.tensor(tangent_v, dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            # Initialize random tangent vectors if not in file
            N = xyz.shape[0]
            tangent_u, tangent_v = self.create_random_orthogonal_tangents_batch(N)
            self._tangent_u = nn.Parameter(tangent_u.requires_grad_(True))
            self._tangent_v = nn.Parameter(tangent_v.requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # FIX: Handle tangent vectors in pruning
        if hasattr(self, '_tangent_u') and self._tangent_u.numel() > 0:
            self._tangent_u = optimizable_tensors["tangent_u"]
            self._tangent_v = optimizable_tensors["tangent_v"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tangent_u=None, new_tangent_v=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        
        # FIX: Add tangent vectors to densification if provided
        if new_tangent_u is not None and new_tangent_v is not None:
            d["tangent_u"] = new_tangent_u
            d["tangent_v"] = new_tangent_v

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        if "tangent_u" in optimizable_tensors:
            self._tangent_u = optimizable_tensors["tangent_u"]
            self._tangent_v = optimizable_tensors["tangent_v"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # FIX: Handle tangent vectors in splitting
        new_tangent_u = None
        new_tangent_v = None
        if hasattr(self, '_tangent_u') and self._tangent_u.numel() > 0:
            new_tangent_u = self._tangent_u[selected_pts_mask].repeat(N,1)
            new_tangent_v = self._tangent_v[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tangent_u, new_tangent_v)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        # FIX: Handle tangent vectors in cloning
        new_tangent_u = None
        new_tangent_v = None
        if hasattr(self, '_tangent_u') and self._tangent_u.numel() > 0:
            new_tangent_u = self._tangent_u[selected_pts_mask]
            new_tangent_v = self._tangent_v[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tangent_u, new_tangent_v)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_from_depth_propagation(self, viewpoint_cam, propagated_depth, filter_mask, gt_image):
        """Standard depth propagation densification"""
        K = viewpoint_cam.K
        cam2world = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()

        height, width = propagated_depth.shape
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        coordinates = torch.stack([x.to(propagated_depth.device), y.to(propagated_depth.device), torch.ones_like(propagated_depth)], dim=-1)
        coordinates = coordinates.view(-1, 3).to(K.device).to(torch.float32)
        coordinates_3D = (K.inverse() @ coordinates.T).T

        coordinates_3D *= propagated_depth.view(-1, 1)
        world_coordinates_3D = (cam2world[:3, :3] @ coordinates_3D.T).T + cam2world[:3, 3]

        world_coordinates_3D = world_coordinates_3D.view(height, width, 3)
        world_coordinates_3D_downsampled = world_coordinates_3D[::8, ::8]
        filter_mask_downsampled = filter_mask[::8, ::8]
        gt_image_downsampled = gt_image.permute(1, 2, 0)[::8, ::8]

        world_coordinates_3D_downsampled = world_coordinates_3D_downsampled[filter_mask_downsampled]
        color_downsampled = gt_image_downsampled[filter_mask_downsampled]

        if world_coordinates_3D_downsampled.shape[0] == 0:
            return

        fused_point_cloud = world_coordinates_3D_downsampled
        fused_color = RGB2SH(color_downsampled)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).to(fused_color.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        original_point_cloud = self.get_xyz
        fused_shape = fused_point_cloud.shape[0]
        all_point_cloud = torch.concat([fused_point_cloud, original_point_cloud], dim=0)
        all_dist2 = torch.clamp_min(distCUDA2(all_point_cloud), 0.0000001)
        dist2 = all_dist2[:fused_shape]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Initialize tangent vectors for new points
        new_tangent_u, new_tangent_v = self.create_random_orthogonal_tangents_batch(fused_shape)

        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        new_tangent_u = nn.Parameter(new_tangent_u.requires_grad_(True))
        new_tangent_v = nn.Parameter(new_tangent_v.requires_grad_(True))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tangent_u, new_tangent_v)

    def enhanced_densify_from_depth_propagation(self, viewpoint_cam, propagated_depth, confidence_map, gt_image):
        """Enhanced densification using both depth and surface information"""
        
        existing_normals = self.get_normals()
        existing_positions = self.get_xyz
        existing_opacities = self.get_opacity
        
        K = viewpoint_cam.K
        cam2world = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
        height, width = propagated_depth.shape
        
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        coordinates = torch.stack([x.to(propagated_depth.device), y.to(propagated_depth.device), 
                                torch.ones_like(propagated_depth)], dim=-1)
        coordinates = coordinates.view(-1, 3).to(K.device).to(torch.float32)
        coordinates_3D = (K.inverse() @ coordinates.T).T
        coordinates_3D *= propagated_depth.view(-1, 1)
        world_coordinates_3D = (cam2world[:3, :3] @ coordinates_3D.T).T + cam2world[:3, 3]
        
        world_coordinates_3D = world_coordinates_3D.view(height, width, 3)
        
        high_conf_mask = confidence_map > 0.8
        medium_conf_mask = (confidence_map > 0.5) & (confidence_map <= 0.8)
        
        combined_mask = high_conf_mask | medium_conf_mask
        if combined_mask.sum() < 50:
            print("Insufficient confident points for densification")
            return
        
        selected_points = []
        selected_colors = []
        selected_confidences = []
        
        if high_conf_mask.sum() > 0:
            high_conf_coords = world_coordinates_3D[::4, ::4][high_conf_mask[::4, ::4]]
            high_conf_colors = gt_image.permute(1, 2, 0)[::4, ::4][high_conf_mask[::4, ::4]]
            high_conf_values = confidence_map[::4, ::4][high_conf_mask[::4, ::4]]
            
            selected_points.append(high_conf_coords)
            selected_colors.append(high_conf_colors)
            selected_confidences.append(high_conf_values)
        
        if medium_conf_mask.sum() > 0:
            medium_conf_coords = world_coordinates_3D[::8, ::8][medium_conf_mask[::8, ::8]]
            medium_conf_colors = gt_image.permute(1, 2, 0)[::8, ::8][medium_conf_mask[::8, ::8]]
            medium_conf_values = confidence_map[::8, ::8][medium_conf_mask[::8, ::8]]
            
            selected_points.append(medium_conf_coords)
            selected_colors.append(medium_conf_colors)
            selected_confidences.append(medium_conf_values)
        
        if len(selected_points) == 0:
            return
            
        fused_point_cloud = torch.cat(selected_points, dim=0)
        fused_colors = torch.cat(selected_colors, dim=0)
        point_confidences = torch.cat(selected_confidences, dim=0)
        
        N = fused_point_cloud.shape[0]
        print(f"Creating {N} new surfels from confident propagations")
        
        fused_color_sh = RGB2SH(fused_colors)
        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).to(fused_point_cloud.device)
        features[:, :3, 0] = fused_color_sh
        features[:, 3:, 1:] = 0.0
        
        if existing_positions.shape[0] > 0:
            all_point_cloud = torch.cat([fused_point_cloud, existing_positions], dim=0)
            all_dist2 = torch.clamp_min(distCUDA2(all_point_cloud), 0.0000001)
            dist2 = all_dist2[:N]
        else:
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        
        # Surface-aware tangent initialization
        tangent_u_list = []
        tangent_v_list = []
        
        for i, new_point in enumerate(fused_point_cloud):
            nearby_surfels = self.find_nearby_surfels(new_point, existing_positions, radius=1.0)
            
            if len(nearby_surfels) > 0 and existing_normals.shape[0] > 0:
                avg_normal = torch.mean(existing_normals[nearby_surfels], dim=0)
                avg_normal = F.normalize(avg_normal, dim=-1)
                tangent_u, tangent_v = self.create_orthogonal_tangents(avg_normal)
            else:
                tangent_u, tangent_v = self.create_random_orthogonal_tangents()
            
            tangent_u_list.append(tangent_u)
            tangent_v_list.append(tangent_v)
        
        # FIX: Move tensor creation outside the loop
        tangent_u_tensor = torch.stack(tangent_u_list)
        tangent_v_tensor = torch.stack(tangent_v_list)
        
        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        
        base_opacity = 0.3
        confidence_bonus = point_confidences * 0.5
        final_opacity = base_opacity + confidence_bonus
        final_opacity = torch.clamp(final_opacity, 0.05, 0.9)
        opacities = inverse_sigmoid(final_opacity.unsqueeze(-1))
        
        curvature_scales = self.estimate_surface_curvature_scaling(
            fused_point_cloud, existing_positions, point_confidences
        )
        scales = scales * curvature_scales.unsqueeze(-1)
        
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        new_tangent_u = nn.Parameter(tangent_u_tensor.requires_grad_(True))
        new_tangent_v = nn.Parameter(tangent_v_tensor.requires_grad_(True))
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                new_opacity, new_scaling, new_rotation,
                                new_tangent_u, new_tangent_v)
        
        print(f"Successfully added {N} new surfels with surface-aware initialization")

    # Helper methods
    def create_random_orthogonal_tangents_batch(self, N):
        """Create N pairs of random orthogonal tangent vectors"""
        tangent_u = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        tangent_u = F.normalize(tangent_u, dim=-1)
        
        tangent_v_raw = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        tangent_v = tangent_v_raw - torch.sum(tangent_v_raw * tangent_u, dim=-1, keepdim=True) * tangent_u
        tangent_v = F.normalize(tangent_v, dim=-1)
        
        return tangent_u, tangent_v

    def find_nearby_surfels(self, query_point, existing_positions, radius=1.0):
        """Find indices of existing surfels within radius of query point"""
        if existing_positions.shape[0] == 0:
            return []
        
        distances = torch.norm(existing_positions - query_point.unsqueeze(0), dim=1)
        nearby_indices = torch.where(distances < radius)[0]
        return nearby_indices

    def create_orthogonal_tangents(self, normal):
        """Create two orthogonal tangent vectors given a normal vector"""
        normal = F.normalize(normal, dim=-1)
        
        ref_vector = torch.tensor([1, 0, 0], device=normal.device, dtype=normal.dtype)
        if torch.abs(torch.dot(normal, ref_vector)) > 0.9:
            ref_vector = torch.tensor([0, 1, 0], device=normal.device, dtype=normal.dtype)
        
        tangent_u = ref_vector - torch.dot(ref_vector, normal) * normal
        tangent_u = F.normalize(tangent_u, dim=-1)
        
        tangent_v = torch.cross(normal, tangent_u, dim=-1)
        tangent_v = F.normalize(tangent_v, dim=-1)
        
        return tangent_u, tangent_v

    def create_random_orthogonal_tangents(self):
        """Create random orthogonal tangent vectors"""
        tangent_u = F.normalize(torch.randn(3, device="cuda"), dim=-1)
        tangent_v = F.normalize(torch.randn(3, device="cuda"), dim=-1)
        
        tangent_v = tangent_v - torch.dot(tangent_v, tangent_u) * tangent_u
        tangent_v = F.normalize(tangent_v, dim=-1)
        
        return tangent_u, tangent_v

    def estimate_surface_curvature_scaling(self, new_points, existing_points, confidences):
        """Estimate scaling factors based on local surface curvature and confidence"""
        N = new_points.shape[0]
        scale_factors = torch.ones(N, device=new_points.device)
        
        if existing_points.shape[0] == 0:
            return scale_factors
        
        for i, point in enumerate(new_points):
            distances = torch.norm(existing_points - point.unsqueeze(0), dim=1)
            
            k = min(8, existing_points.shape[0])
            _, nearest_indices = torch.topk(distances, k, largest=False)
            nearest_points = existing_points[nearest_indices]
            
            if k >= 3:
                center = torch.mean(nearest_points, dim=0)
                deviations = torch.norm(nearest_points - center.unsqueeze(0), dim=1)
                surface_variation = torch.std(deviations)
                
                confidence_factor = confidences[i].item()
                variation_factor = torch.clamp(surface_variation, 0.1, 2.0)
                
                scale_factors[i] = variation_factor * (2.0 - confidence_factor)
                scale_factors[i] = torch.clamp(scale_factors[i], 0.5, 3.0)
        
        return scale_factors

    def project_point_to_pixel(self, point_3d, viewpoint_cam):
        """Project a 3D point to 2D pixel coordinates"""
        try:
            # Transform to camera coordinates
            w2c = viewpoint_cam.world_view_transform.transpose(0, 1)
            point_cam = w2c[:3, :3] @ point_3d + w2c[:3, 3]
            
            # Project to image plane
            K = viewpoint_cam.K
            point_2d = K @ point_cam
            
            if point_2d[2] <= 0:  # Behind camera
                return None
                
            pixel = point_2d[:2] / point_2d[2]
            return pixel
        except:
            return None