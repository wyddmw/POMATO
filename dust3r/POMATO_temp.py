# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.heads import head_factory
from dust3r.utils.misc import transpose_to_landscape, freeze_all_params  # noqa
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud

inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


'''
Note: Our model was originally built on top of MASt3R (lol). With the release of MonST3R, 
which offers improved capabilities for dynamic scenario reconstruction, we switched to 
using its checkpoint for training, while still leveraging MASt3R's framework.

However, MASt3R's codebase is not particularly beginner-friendly. In this repo,
we simply put the core parts of our model and integrate it into MonST3R's framework for inference and evaluation.
'''

class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, freeze='none', **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.temporal_length = kwargs.pop('temporal_length')
        
        super().__init__(**kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode

        # utilize original dust3r head without local feature desc
        self.downstream_head1 = head_factory(head_type='dpt_temp', output_mode='pts3d', net=self, has_conf=bool(conf_mode), temp_length=self.temporal_length)
        self.downstream_head2 = head_factory(head_type='dpt_temp', output_mode='pts3d', net=self, has_conf=bool(conf_mode), temp_length=self.temporal_length)
        self.downstream_head3 = head_factory(head_type='dpt_temp', output_mode='pts3d', net=self, has_conf=bool(conf_mode), temp_length=self.temporal_length)
        
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        self.head3 = transpose_to_landscape(self.downstream_head3, activate=landscape_only)

    def forward(self, view1, view2=None, inference=False, mode='tracking'):
        assert view2 == None, print("view2 should be empty for temporal inference")
        view2 = {}        
        if not inference:
            mode = np.random.choice(['tracking', 'fast_recon', 'video_depth'])

        # define the keyframe index
        if mode == 'fast_recon':
            keyframe_index = self.temporal_length - 1

        elif mode == 'video_depth':
            keyframe_index = 0
            
        elif mode == 'tracking':
            keyframe_index = 0

        for key, data in view1.items():
            if key == 'img':
                continue
            if mode == 'fast_recon':
                # 3D reconstruction, the keyframe is set to the last frame within a temporal window.
                # keyframe_index = self.temporal_length - 1
                data_repeat_shape = torch.tensor(data.size())
                view2[key] = data.reshape(-1, *data_repeat_shape[2:])
                view1[key] = data[:, keyframe_index:keyframe_index+1].expand(*data_repeat_shape).contiguous().reshape(-1, *data_repeat_shape[2:])
            elif mode == 'video_depth':
                data_repeat_shape = torch.tensor(data.size())
                view2[key] = data.reshape(-1, *data_repeat_shape[2:])
                view1[key] = data.reshape(-1, *data_repeat_shape[2:])
            else:
                # 3D point tracking, the keyframe is the first frame within a temporal window.
                # keyframe_index = 0
                data_repeat_shape = torch.tensor(data.size())
                view1[key] = data.reshape(-1, *data_repeat_shape[2:])
                view2[key] = data[:, keyframe_index:keyframe_index+1].expand(*data_repeat_shape).contiguous().reshape(-1, *data_repeat_shape[2:])
        B, T, _, H, W = view1['img'].shape
        assert self.temporal_length == T
        true_shape = view1['img'].shape[-2:]
        true_shape = torch.tensor([H, W])[None].repeat(B*T, 1)
        view1['img'] = view1['img'].view(B*T, 3, H, W)

        out, pos, _ = self._encode_image(view1['img'], true_shape)
        token_num, token_dim = out.shape[-2:]
        # if mode == 'fast_recon' or mode == 'tracking':
        #     assert keyframe_index is not None, 'keyframe_index should be provided'

        out = out.reshape(B, T, token_num, token_dim)
        if mode == 'fast_recon':
            feat2 = out.view(-1, token_num, token_dim)
            feat1 = out[:, keyframe_index:keyframe_index+1].repeat(1, T, 1, 1).contiguous()
            feat1 = feat1.view(-1, token_num, token_dim)
        elif mode == 'video_depth':
            feat2 = out.view(-1, token_num, token_dim)
            feat1 = out.view(-1, token_num, token_dim)
        else:
            feat1 = out.view(-1, token_num, token_dim)
            feat2 = out[:, keyframe_index:keyframe_index+1].repeat(1, T, 1, 1).contiguous()
            feat2 = feat2.view(-1, token_num, token_dim)
    
        # combine all ref images into object-centric representation
        with torch.no_grad():
            dec1, dec2 = self._decoder(feat1, pos, feat2, pos)    
        
        # if not train:
        with torch.cuda.amp.autocast(enabled=inference):
            res1, _ = self._downstream_head(1, [tok.float() for tok in dec1], true_shape)
            res2, _ = self._downstream_head(2, [tok.float() for tok in dec2], true_shape)
            res3, _ = self._downstream_head(3, [tok.float() for tok in dec2], true_shape)

        res3['mono_depth'] = res1['pts3d']
        res3['mono_conf'] = res1['conf']
        res3['pts3d_matching'] = res3['pts3d']          # [B*T, H, W, 3]
        res3['conf_matching'] = res3['conf']
        res3['recon_pts3d'] = res2['pts3d']             # predict view2's pts3d in view1's frame
        res3['recon_conf'] = res2['conf']               # predict view2's pts3d in view1's frame
        res2['pts3d_in_other_view'] = res2['pts3d']     # predict view2's pts3d in view1's frame

        if 'pts3d' in view1.keys():
            # NOTE: Here is the temporal reconstruction loss we use for training.
            matching_loss = temporal_recon_loss(view1, view2, res1, res2, res3, self.temporal_length, keyframe_index)
            res3.update({'dynamic_matching_loss': matching_loss})            
        
        return res3
    

def temporal_recon_loss(view1, view2, pred1=None, pred2=None, pred3=None, temporal_length=None, matching_pair_index=None):
    spatial_shape = view1['img'].shape[-2:]
    num_points = spatial_shape[0] * spatial_shape[1]
    device = view1['img'].device
    in_camera1 = inv(view1['camera_pose'])                  # camera_to_world -> world_to_camera
    
    # compute the pts3d loss first with vanilla loss in Dust3R
    gt_pts1 = geotrf(in_camera1, view1['pts3d'])
    gt_pts2 = geotrf(in_camera1, view2['pts3d'])            # project to the coordinate of the first view1
    
    pred_pts1 = pred1['pts3d']
    pred_conf1 = pred1['conf']
    pred_pts2 = pred2['pts3d_in_other_view']
    pred_conf2 = pred2['conf']
    valid1 = view1['valid_mask']
    valid2 = view2['valid_mask']
    pts3d_matching = pred3['pts3d_matching'].reshape(-1, temporal_length, *spatial_shape, 3)
    conf_matching = pred3['conf_matching'].reshape(-1, temporal_length, *spatial_shape)
    batch_size = pts3d_matching.shape[0]
    # pointcloud normalization for temporal
    gt_pts1_norm, gt_pts2_norm = normalize_pointcloud(gt_pts1, gt_pts2, valid1=valid1, valid2=valid2, temporal_length=temporal_length)      # [B*T, H, W, 3]
    pr_pts1_norm, pr_pts2_norm, norm_factor_pred_temp = normalize_pointcloud(pred_pts1, pred_pts2, valid1=valid1, valid2=valid2, temporal_length=temporal_length, ret_factor=True)
    # normalize matching pts3d for temporal consistency
    pr_pts3_norm_temp = pts3d_matching / norm_factor_pred_temp
    pr_pts3_norm_temp = pr_pts3_norm_temp.reshape(batch_size, temporal_length, *spatial_shape, 3)

    loss_pts3d1_temp = torch.norm(gt_pts1_norm[valid1] - pr_pts1_norm[valid1], dim=-1)
    loss_pts3d2_temp = torch.norm(gt_pts2_norm[valid2] - pr_pts2_norm[valid2], dim=-1)
    conf_loss1_temp = loss_pts3d1_temp * pred_conf1[valid1] - 0.2 * torch.log(pred_conf1[valid1])
    conf_loss2_temp = loss_pts3d2_temp * pred_conf2[valid2] - 0.2 * torch.log(pred_conf2[valid2])

    # pointcloud normalization for per frame
    gt_pts1_norm, gt_pts2_norm = normalize_pointcloud(gt_pts1, gt_pts2, valid1=valid1, valid2=valid2)      # [B*T, H, W, 3]
    pr_pts1_norm, pr_pts2_norm, norm_factor_pred_perframe = normalize_pointcloud(pred_pts1, pred_pts2, valid1=valid1, valid2=valid2, ret_factor=True)
    pr_pts3_norm_perframe = pts3d_matching.flatten(0,1) / norm_factor_pred_perframe
    pr_pts3_norm_perframe = pr_pts3_norm_perframe.reshape(batch_size, temporal_length, *spatial_shape, 3)
    
    loss_pts3d1_perframe = torch.norm(gt_pts1_norm[valid1] - pr_pts1_norm[valid1], dim=-1)
    loss_pts3d2_perframe = torch.norm(gt_pts2_norm[valid2] - pr_pts2_norm[valid2], dim=-1)
    conf_loss1_perframe = loss_pts3d1_perframe * pred_conf1[valid1] - 0.2 * torch.log(pred_conf1[valid1])
    conf_loss2_perframe = loss_pts3d2_perframe * pred_conf2[valid2] - 0.2 * torch.log(pred_conf2[valid2])
    # compute the pts3d and conf loss for both temporal and perframe    
    loss_pts3d1 = loss_pts3d1_perframe.mean() + loss_pts3d1_temp.mean() * 3
    loss_conf1 = conf_loss1_perframe.mean() + conf_loss1_temp.mean()
    loss_pts3d2 = loss_pts3d2_perframe.mean() + loss_pts3d2_temp.mean() * 3
    loss_conf2 = conf_loss2_perframe.mean() + conf_loss2_temp.mean()

    dense_matching_loss = sparse_matching_loss = 0
    dense_conf_loss = sparse_conf_loss = 0
    loss_pts3d = torch.tensor([0])

    if view2['sparse'].any() == False:
        dense_batch_index = view2['sparse'].reshape(batch_size, temporal_length).all(dim=1) == False      # [B, 2]
        gt_pts2_norm = gt_pts2_norm.reshape(batch_size, -1, *spatial_shape, 3)[dense_batch_index]
        dense_pred_pts3_norm_temp = pr_pts3_norm_temp[dense_batch_index]
        dense_pred_pts3_norm_perframe = pr_pts3_norm_perframe[dense_batch_index]
        dense_conf_pred = conf_matching[dense_batch_index]

        valid_flow_mask_view1 = (view2['valid_mask'] * view2['valid_flow_mask']).reshape(batch_size, temporal_length, *spatial_shape)[dense_batch_index]
        dense_matching_loss_temp = torch.norm(dense_pred_pts3_norm_temp[valid_flow_mask_view1] - gt_pts2_norm[valid_flow_mask_view1], dim=-1)
        dense_matching_loss_perframe = torch.norm(dense_pred_pts3_norm_perframe[valid_flow_mask_view1] - gt_pts2_norm[valid_flow_mask_view1], dim=-1)
        dense_conf_loss_temp = (dense_matching_loss_temp * dense_conf_pred[valid_flow_mask_view1] - 0.2 * torch.log(dense_conf_pred[valid_flow_mask_view1])).mean()
        dense_conf_loss_perframe = (dense_matching_loss_perframe * dense_conf_pred[valid_flow_mask_view1] - 0.2 * torch.log(dense_conf_pred[valid_flow_mask_view1])).mean()
        dense_matching_loss = dense_matching_loss_temp.mean() + dense_matching_loss_perframe.mean()
        dense_conf_loss = dense_conf_loss_temp.mean() + dense_conf_loss_perframe.mean()

    # update loss for sparse tracking
    if view2['sparse'].any() == True:
        gt_pts1_norm = gt_pts1_norm.reshape(-1, temporal_length, *spatial_shape, 3)
        sparse_batch_index = view1['sparse'].reshape(-1, temporal_length).all(dim=1) == True      # [B, 2]
        # supervise the view0 -> view1 as visible mask is obtained according to the view0
        sparse_pts3d_matching_temp = pr_pts3_norm_temp[sparse_batch_index].reshape(-1, num_points, 3)                 # [B_sparse*T, H*W, 2], img0
        sparse_pts3d_matching_perframe = pr_pts3_norm_perframe[sparse_batch_index].reshape(-1, num_points, 3)                 # [B_sparse*T, H*W, 2], img0

        sparse_conf_matching = conf_matching[sparse_batch_index].reshape(-1, num_points)
        sparse_pts3d_matching_gt = gt_pts1_norm[sparse_batch_index].reshape(-1, num_points, 3)
        sparse_visible_mask = view1['visibs'].reshape(batch_size, temporal_length, -1)[sparse_batch_index]
        sparse_valids_mask = view1['valids'].reshape(batch_size, temporal_length, -1)[sparse_batch_index]
        tar_points = view1['trajs_2d'].round().reshape(batch_size, temporal_length, -1, 2)[sparse_batch_index]  # [B_sparse, T, num_trajs, 2] 
        ref_points = view2['trajs_2d'].round().reshape(batch_size, temporal_length, -1, 2)[sparse_batch_index]  # [B_sparse, T, num_trajs, 2] 
        
        ref_index = ref_points[..., 0] + ref_points[..., 1] * spatial_shape[1]
        tar_index = tar_points[..., 0] + tar_points[..., 1] * spatial_shape[1]
        ref_index = torch.clamp(ref_index, 0, num_points-1).long().flatten(0, 1)
        tar_index = torch.clamp(tar_index, 0, num_points-1).long().flatten(0, 1)
        ref_pts_preds_temp = torch.gather(sparse_pts3d_matching_temp, 1, ref_index[..., None].expand(-1, -1, 3))
        ref_pts_preds_perframe = torch.gather(sparse_pts3d_matching_perframe, 1, ref_index[..., None].expand(-1, -1, 3))

        tar_pts_gts = torch.gather(sparse_pts3d_matching_gt, 1, tar_index[..., None].expand(-1, -1, 3))
        sparse_conf_matching = torch.gather(sparse_conf_matching, 1, ref_index)
        valid_index = sparse_valids_mask & sparse_visible_mask
        valid_index = valid_index.flatten(0, 1)

        sparse_matching_loss_temp = torch.norm(ref_pts_preds_temp[valid_index] - tar_pts_gts[valid_index], dim=-1)
        sparse_matching_loss_perframe = torch.norm(ref_pts_preds_perframe[valid_index] - tar_pts_gts[valid_index], dim=-1)

        sparse_conf_loss_temp = (sparse_matching_loss_temp * sparse_conf_matching[valid_index] - 0.2 * torch.log(sparse_conf_matching[valid_index])).mean()
        sparse_conf_loss_perframe = (sparse_matching_loss_perframe * sparse_conf_matching[valid_index] - 0.2 * torch.log(sparse_conf_matching[valid_index])).mean()

        sparse_matching_loss = sparse_matching_loss_temp.mean() + sparse_matching_loss_perframe.mean()
        sparse_conf_loss = sparse_conf_loss_temp.mean() + sparse_conf_loss_perframe.mean()
        
    loss_pts3d = sparse_matching_loss + dense_matching_loss
    loss_matching_conf = sparse_conf_loss + dense_conf_loss
    if torch.isnan(loss_pts3d) or loss_pts3d == 0:      # no valid matching points or dense matching
        # directly set the loss to zero
        loss_pts3d = pred3['pts3d_matching'].mean() * 0 
        loss_matching_conf = pred3['conf_matching'].mean() * 0
    if torch.isnan(loss_pts3d1):
        loss_pts3d1 = pred_pts1.mean() * 0
        loss_conf1 = pred_conf1.mean() * 0        
    
    if torch.isnan(loss_pts3d2):
        loss_pts3d2 = pred_pts2.mean() * 0
        loss_conf2 = pred_conf2.mean() * 0
    
    loss_dict = {
        'loss_pts3d': loss_pts3d,
        'loss_conf': loss_matching_conf,
        'loss_pts3d1': loss_pts3d1,
        'loss_pts3d2': loss_pts3d2,
        'conf_loss1': loss_conf1,
        'conf_loss2': loss_conf2
    }

    return loss_dict