import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib

from dust3r.cloud_opt.base_opt import BasePCOptimizer, edge_str
from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.utils.geometry import xy_grid, geotrf, depthmap_to_pts3d
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.goem_opt import DepthBasedWarping, OccMask, WarpImage, depth_regularization_si_weighted, tum_to_pose_matrix

from third_party.raft import load_RAFT
from dust3r.cloud_opt.flow_viz import save_vis_flow_tofile
# from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.):
    loss_raw_shape = F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='none')
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask
    return torch.sum(loss_raw_shape * per_pixel_mask) / torch.sum(per_pixel_mask)

def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / torch.sum(mask)
    return v  # , v.item()

class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, shared_focal=False, flow_loss_fn='smooth_l1', flow_loss_weight=0.0, 
                 depth_regularize_weight=0.0, num_total_iter=300, temporal_smoothing_weight=0, translation_weight=0.1, flow_loss_start_epoch=0.15, flow_loss_thre=50,
                 sintel_ckpt=False, use_self_mask=False, pxl_thre=50, sam2_mask_refine=False, moving_distance_thres_global=0.1, moving_distance_thres_adjacent=0.35, adjacent_intervals=10, reproj_focal=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break
        self.num_total_iter = num_total_iter
        self.temporal_smoothing_weight = temporal_smoothing_weight
        self.translation_weight = translation_weight
        self.flow_loss_flag = False
        self.flow_loss_start_epoch = flow_loss_start_epoch
        self.flow_loss_thre = flow_loss_thre
        self.optimize_pp = optimize_pp
        self.pxl_thre = pxl_thre
        self.moving_distance_thres_global = moving_distance_thres_global
        self.moving_distance_thres_adjacent = moving_distance_thres_adjacent
        self.set_device()
        # fix random seed here
        self.set_seed(42)
        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.shared_focal = shared_focal
        self.reproj_focal = reproj_focal
        if self.reproj_focal:
            # compute 
            pass
        else:
            if self.shared_focal:
                self.im_focals = nn.ParameterList(torch.FloatTensor(
                    [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes[:1])  # camera intrinsics
            else:
                self.im_focals = nn.ParameterList(torch.FloatTensor(
                    [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area) #(num_imgs, H*W)

        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer('_weight_i', ParameterStack(
            [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j', ParameterStack(
            [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j_matching', ParameterStack(
            [self.conf_trf(self.conf_j_matching[i_j]) for i_j in self.str_edges], fill=self.max_area))

        # precompute all the things
        self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])
        
        self.adjacent_pred_j = ParameterStack(self.pred_j, self.adjacent_edges, fill=self.max_area)
        self.adjacent_pred_m = ParameterStack(self.pred_m, self.adjacent_edges, fill=self.max_area)
        self.adjacent_pairs_ij = (self._ei - self._ej).abs() < adjacent_intervals
        self.adjacent_pairs_ij = self.adjacent_pairs_ij[:, None, None, None].cuda()
        self.adjacent_pairs_ji = (self._ej - self._ei).abs() < adjacent_intervals
        self.adjacent_pairs_ji = self.adjacent_pairs_ji[:, None, None, None].cuda()
        self.depth_wrapper = DepthBasedWarping()
        self.backward_warper = WarpImage()
        self.depth_regularizer = depth_regularization_si_weighted
        if flow_loss_fn == 'smooth_l1':
            self.flow_loss_fn = smooth_L1_loss_fn
        elif flow_loss_fn == 'mse':
            self.low_loss_fn = mse_loss_fn

        self.flow_loss_weight = flow_loss_weight
        self.depth_regularize_weight = depth_regularize_weight
        if self.flow_loss_weight > 0:
            if use_self_mask: self.get_motion_mask_from_pairs(*args)
            # use RAFT optical flow
            # self.flow_ij, self.flow_ji, self.flow_valid_mask_i, self.flow_valid_mask_j = self.get_flow(sintel_ckpt) # (num_pairs, 2, H, W)
    
    def _cal_pred_intrinsic(self, sample_pair_nums=10):
        pts3d_pred_list = []
        conf_pred_list = []
        for key, pts3d in self.pred_i.items():
            # if i < sample_pair_nums:
            pts3d_pred_list.append(pts3d)
            conf_pred_list.append(self.conf_i[key])
            if len(pts3d_pred_list) > sample_pair_nums:
                break

        H, W = pts3d_pred_list[0].shape[:2]
        cx = W // 2
        cy = H // 2
        x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        x_coords = torch.from_numpy(x_coords).to(self.flow_ij_device)
        y_coords = torch.from_numpy(y_coords).to(self.flow_ij_device)

        def _calculate_intrinsic(pts, conf, x_coords, y_coords, cx, cy, percentile=0.5):
            mask = conf > torch.quantile(conf.flatten(), percentile)
            fx = (x_coords - cx) / (pts[..., 0] / (pts[..., 2] + 1e-8))
            fy = (y_coords - cy) / (pts[..., 1] / (pts[..., 2] + 1e-8))
            mask_fx, mask_fy = fx[mask], fy[mask]
            result_fx, result_fy = mask_fx.nanmedian().float(), mask_fy.nanmedian().float()
            return result_fx, result_fy
        
        pred_fx, pred_fy = _calculate_intrinsic(pts3d_pred_list, conf_pred_list, x_coords, y_coords, cx, cy)
        return pred_fx, pred_fy
        
    def set_device(self):
        self.flow_ij_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_flow(self, sintel_ckpt=False): #TODO: test with gt flow
        print('precomputing flow...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_valid_flow_mask = OccMask(th=3.0)
        pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]
        flow_net = load_RAFT() if sintel_ckpt else load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        flow_net = flow_net.to(device)
        flow_net.eval()

        with torch.no_grad():
            chunk_size = 12
            flow_ij = []
            flow_ji = []
            num_pairs = len(pair_imgs[0])
            for i in tqdm(range(0, num_pairs, chunk_size)):
                end_idx = min(i + chunk_size, num_pairs)
                imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
                        torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
                flow_ij.append(flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                        imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                        iters=20, test_mode=True)[1])
                flow_ji.append(flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                        imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                        iters=20, test_mode=True)[1])

            flow_ij = torch.cat(flow_ij, dim=0)
            flow_ji = torch.cat(flow_ji, dim=0)
            for i in range(flow_ij.shape[0]):
                save_vis_flow_tofile(flow_ij[i].permute(1, 2, 0).cpu().numpy(), f'flow_viz/flow_ij_{i}.png')
            valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
            valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
        print('flow precomputed')
        # delete the flow net
        if flow_net is not None: del flow_net
        return flow_ij, flow_ji, valid_mask_i, valid_mask_j

    def get_motion_mask_from_pairs(self, view1, view2, pred1, pred2):
        assert self.is_symmetrized, 'only support symmetric case'
        symmetry_pairs_idx = [(i, i+len(self.edges)//2) for i in range(len(self.edges)//2)]
        
        # intrinsics_i_pairs = []
        # intrinsics_j_pairs = []
        # R_i = []
        # R_j = []
        # T_i = []
        # T_j = []
        # depth_maps_i = []
        # depth_maps_j = []
        matching_maps_i = []
        matching_maps_j = []

        depth_proj_i = []
        depth_proj_j = []    
        for i, j in tqdm(symmetry_pairs_idx):
            new_view1 = {}
            new_view2 = {}
            for key in view1.keys():
                if isinstance(view1[key], list):
                    new_view1[key] = [view1[key][i], view1[key][j]]
                    new_view2[key] = [view2[key][i], view2[key][j]]
                elif isinstance(view1[key], torch.Tensor):
                    new_view1[key] = torch.stack([view1[key][i], view1[key][j]])
                    new_view2[key] = torch.stack([view2[key][i], view2[key][j]])
            new_view1['idx'] = [0, 1]
            new_view2['idx'] = [1, 0]
            new_pred1 = {}
            new_pred2 = {}
            for key in pred1.keys():
                if isinstance(pred1[key], list):
                    new_pred1[key] = [pred1[key][i], pred1[key][j]]
                elif isinstance(pred1[key], torch.Tensor):
                    new_pred1[key] = torch.stack([pred1[key][i], pred1[key][j]])
            for key in pred2.keys():
                if isinstance(pred2[key], list):
                    new_pred2[key] = [pred2[key][i], pred2[key][j]]
                elif isinstance(pred2[key], torch.Tensor):
                    new_pred2[key] = torch.stack([pred2[key][i], pred2[key][j]])
            pair_viewer = PairViewer(new_view1, new_view2, new_pred1, new_pred2, verbose=False)
            # intrinsics_i_pairs.append(pair_viewer.get_intrinsics()[0])
            # intrinsics_j_pairs.append(pair_viewer.get_intrinsics()[1])
            # R_i.append(pair_viewer.get_im_poses()[0][:3, :3])
            # R_j.append(pair_viewer.get_im_poses()[1][:3, :3])
            # T_i.append(pair_viewer.get_im_poses()[0][:3, 3:])
            # T_j.append(pair_viewer.get_im_poses()[1][:3, 3:])
            # depth_maps_i.append(pair_viewer.get_depthmaps()[0])
            # depth_maps_j.append(pair_viewer.get_depthmaps()[1])
            matching_maps_i.append(pair_viewer.get_matchingmaps()[0])
            matching_maps_j.append(pair_viewer.get_matchingmaps()[1])
            depth_proj_i.append(pair_viewer.get_depthprojmaps()[0])
            depth_proj_j.append(pair_viewer.get_depthprojmaps()[1])
        
        # intrinsics_i_median = torch.stack(intrinsics_i_pairs).to(self.flow_ij_device)
        # intrinsics_j_median = torch.stack(intrinsics_j_pairs).to(self.flow_ij_device)
        # # update intrinsics with median value
        # intrinsics_i = intrinsics_i_median.median(dim=0)[0]
        # intrinsics_i = intrinsics_i[None].repeat(len(self.edges)//2, 1, 1)
        # intrinsics_j = intrinsics_j_median.median(dim=0)[0]
        # intrinsics_j = intrinsics_j[None].repeat(len(self.edges)//2, 1, 1)

        # R_i = torch.stack(R_i).to(self.flow_ij_device)      # [num_pairs, 3, 3]
        # R_j = torch.stack(R_j).to(self.flow_ij_device)
        # T_i = torch.stack(T_i).to(self.flow_ij_device)
        # T_j = torch.stack(T_j).to(self.flow_ij_device)
        # depth_maps_i = torch.stack(depth_maps_i).unsqueeze(1).to(self.flow_ij_device)
        # depth_maps_j = torch.stack(depth_maps_j).unsqueeze(1).to(self.flow_ij_device)
        
        # update the matching prediction
        matching_maps_i = torch.stack(matching_maps_i).unsqueeze(1).to(self.flow_ij_device)
        matching_maps_j = torch.stack(matching_maps_j).unsqueeze(1).to(self.flow_ij_device)
        B, _, H, W = matching_maps_i.shape[:-1]
        matching_maps_i = matching_maps_i.reshape(B, -1, 3).permute(0, 2, 1)
        matching_maps_j = matching_maps_j.reshape(B, -1, 3).permute(0, 2, 1)
        
        self.matching_maps_ij = torch.cat((matching_maps_i, matching_maps_j), dim=0)       # [num_pairs, 3, H*W]
        self.matching_maps_ji = torch.cat((matching_maps_j, matching_maps_i), dim=0)
        depth_proj_i = torch.stack(depth_proj_i).to(self.flow_ij_device)
        depth_proj_j = torch.stack(depth_proj_j).to(self.flow_ij_device)
        # NOTE: Compute the dynamic map by computing the distance between head3 and head2. Regions with large distance are considered as dynamic.
        matching_maps_i = matching_maps_i.reshape(B, 3, H, W).permute(0, 2, 3, 1)
        matching_maps_j = matching_maps_j.reshape(B, 3, H, W).permute(0, 2, 3, 1)
        distance_map_i_proj = torch.norm(matching_maps_i - depth_proj_i, dim=-1)
        dynamic_mask_i = distance_map_i_proj > distance_map_i_proj.flatten(1,2).mean(dim=-1)[:, None, None] * self.moving_distance_thres_global
        dynamic_mask_i = dynamic_mask_i.reshape(B, H, W)
        distance_map_j_proj = torch.norm(matching_maps_j - depth_proj_j, dim=-1)
        dynamic_mask_j = distance_map_j_proj > distance_map_j_proj.flatten(1,2).mean(dim=-1)[:, None, None] * self.moving_distance_thres_global
        dynamic_mask_j = dynamic_mask_j.reshape(B, H, W)
        
        self.dynamic_mask_ij = torch.cat((dynamic_mask_i, dynamic_mask_j), dim=0)
        self.dynamic_mask_ji = torch.cat((dynamic_mask_j, dynamic_mask_i), dim=0)
        
        # # visualize the reliable pairwise dynamic mask
        # import copy
        # import cv2
        # dynamic_mask_ij_vis = copy.deepcopy(self.dynamic_mask_ij)
        # dynamic_mask_ij_vis = dynamic_mask_ij_vis.data.cpu().numpy().astype(np.uint8) * 255
        # for i in range(self.dynamic_mask_ij.shape[0]):
        #     cv2.imwrite(f'dynamic_mask_ij_vis_{i}.png', dynamic_mask_ij_vis[i])

        adjacent_distance = (self.adjacent_pred_m - self.adjacent_pred_j).norm(dim=-1)
        self.dynamic_masks = adjacent_distance > adjacent_distance.mean(dim=-1)[:, None] * self.moving_distance_thres_adjacent
        self.dynamic_masks = self.dynamic_masks.reshape(self.n_imgs-1, H, W)
        self.dynamic_masks = torch.cat((self.dynamic_masks, torch.zeros((1, H, W))), dim=0).to(self.flow_ij_device)

    def refine_motion_mask_w_sam2(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Save previous TF32 settings
        if device == 'cuda':
            prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        try:
            autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                frame_tensors = torch.from_numpy(np.array((self.imgs))).permute(0, 3, 1, 2).to(device)
                inference_state = predictor.init_state(video_path=frame_tensors)
                mask_list = [self.dynamic_masks[i] for i in range(self.n_imgs)]
                
                ann_obj_id = 1
                self.sam2_dynamic_masks = [[] for _ in range(self.n_imgs)]

                # Process even frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 1:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 0:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Process odd frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 0:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 1:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
                    
                # Update dynamic masks
                for i in range(self.n_imgs):
                    self.sam2_dynamic_masks[i] = torch.from_numpy(self.sam2_dynamic_masks[i][0]).to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i].to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i] | self.sam2_dynamic_masks[i]
                
                # Clean up
                del predictor
        finally:
            # Restore previous TF32 settings
            if device == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
                torch.backends.cudnn.allow_tf32 = prev_allow_cudnn_tf32

    def _check_all_imgs_are_selected(self, msk):
        self.msk = torch.from_numpy(np.array(msk, dtype=bool)).to(self.device)
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'
        pass

    def preset_pose(self, known_poses, pose_msk=None, requires_grad=False):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        if known_poses.shape[-1] == 7: # xyz wxyz
            known_poses = [tum_to_pose_matrix(pose) for pose in known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)
        if len(known_poses) == self.n_imgs:
            if requires_grad:
                self.im_poses.requires_grad_(True)
            else:
                self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_intrinsics(self, known_intrinsics, msk=None):
        if isinstance(known_intrinsics, torch.Tensor) and known_intrinsics.ndim == 2:
            known_intrinsics = [known_intrinsics]
        for K in known_intrinsics:
            assert K.shape == (3, 3)
        self.preset_focal([K.diagonal()[:2].mean() for K in known_intrinsics], msk)
        if self.optimize_pp:
            self.preset_principal_point([K[:2, 2] for K in known_intrinsics], msk)

    def preset_focal(self, known_focals, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))
        if len(known_focals) == self.n_imgs:
            if requires_grad:
                self.im_focals.requires_grad_(True)
            else:
                self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.im_focals[0]] * self.n_imgs, dim=0)
        else:
            log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param
    
    def preset_depthmap(self, known_depthmaps, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, depth in zip(self._get_msk_indices(msk), known_depthmaps):
            if self.verbose:
                print(f' (setting depthmap #{idx})')
            self._no_grad(self._set_depthmap(idx, depth))

        if len(known_depthmaps) == self.n_imgs:
            if requires_grad:
                self.im_depthmaps.requires_grad_(True)
            else:
                self.im_depthmaps.requires_grad_(False)
    
    def _set_init_depthmap(self):
        depth_maps = self.get_depthmaps(raw=True)
        self.init_depthmap = [dm.detach().clone() for dm in depth_maps]

    def get_init_depthmaps(self, raw=False):
        res = self.init_depthmap
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def depth_to_pts3d_partial(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps()

        # convert focal to (1,2,H,W) constant field
        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *self.imshapes[i])
        # get pointmaps in camera frame
        rel_ptmaps = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[i:i+1])[0] for i in range(im_poses.shape[0])]
        # project to world frame
        return [geotrf(pose, ptmap) for pose, ptmap in zip(im_poses, rel_ptmaps)]
    
    def get_pts3d(self, raw=False, **kwargs):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def forward(self, epoch=9999):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)
        
        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)      # projected head1
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)      # projected head2
    
        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j
        
        # camera temporal loss
        if self.temporal_smoothing_weight > 0:
            temporal_smoothing_loss = self.relative_pose_loss(self.get_im_poses()[:-1], self.get_im_poses()[1:]).sum()
        else:
            temporal_smoothing_loss = 0
        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch: # enable flow loss after certain epoch
            R_all, T_all = self.get_im_poses()[:,:3].split([3, 1], dim=-1)
            R1, T1 = R_all[self._ei], T_all[self._ei]
            R2, T2 = R_all[self._ej], T_all[self._ej]
            K_all = self.get_intrinsics()
            inv_K_all = torch.linalg.inv(K_all)
            K_1, inv_K_1 = K_all[self._ei], inv_K_all[self._ei]
            K_2, inv_K_2 = K_all[self._ej], inv_K_all[self._ej]
            depth_all = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            depth1, depth2 = depth_all[self._ei], depth_all[self._ej]
            disp_1, disp_2 = 1 / (depth1 + 1e-6), 1 / (depth2 + 1e-6)
            ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K_2, inv_K_1)
            ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K_1, inv_K_2)
            B, _, H, W = ego_flow_1_2.shape
            
            # # RAFT optical flow
            # flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], self.flow_ij[:, :2], ~self.dynamic_mask_ij[:, None], per_pixel_thre=self.pxl_thre)
            # flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], self.flow_ji[:, :2], ~self.dynamic_mask_ij[:, None], per_pixel_thre=self.pxl_thre)
            
            # OURS
            flow_ij = torch.matmul(K_all[0:1], self.matching_maps_ij / (self.matching_maps_ij[:, -1:, ...] + 1e-6)) - self.depth_wrapper.coord
            flow_ij = flow_ij[:, :2]
            flow_ij = flow_ij.reshape(-1, 2, H, W)
            flow_ji = torch.cat((flow_ij[len(self.edges)//2:], flow_ij[:len(self.edges)//2]), dim=0)
            flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], flow_ij[:, :2].detach(), ~self.dynamic_mask_ij[:, None], per_pixel_thre=self.pxl_thre)
            flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], flow_ji[:, :2].detach(), ~self.dynamic_mask_ji[:, None], per_pixel_thre=self.pxl_thre)
            flow_loss = flow_loss_i + flow_loss_j
            print(f'flow loss: {flow_loss.item()}')
            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0: 
                flow_loss = 0
                self.flow_loss_flag = True
            # # flow visualization
            #     for i in range(self.flow_ij.shape[0]):
            #         save_vis_flow_tofile(flow_ij[:, :2].detach()[i].permute(1, 2, 0).cpu().numpy(), f'flow_viz_ours_frame10/flow_ij_{i}.png')
            
        else:    
            flow_loss = 0
        
        if self.depth_regularize_weight > 0:
            init_depthmaps = torch.stack(self.get_init_depthmaps(raw=False)).unsqueeze(1)
            depthmaps = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            depth_prior_loss = self.depth_regularizer(depthmaps, init_depthmaps, dynamic_masks_all)
        else:
            depth_prior_loss = 0
            
        # self.flow_loss_weight = 0.0
        loss = (li + lj) * 1 + self.temporal_smoothing_weight * temporal_smoothing_loss + \
                self.flow_loss_weight * flow_loss + self.depth_regularize_weight * depth_prior_loss
                
        return loss

    def relative_pose_loss(self, RT1, RT2):
        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss + translation_loss * self.translation_weight
        return pose_loss

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img

def ordered_ratio(disp_a, disp_b, mask=None):
    ratio_a = torch.maximum(disp_a, disp_b) / \
        (torch.minimum(disp_a, disp_b)+1e-5)
    if mask is not None:
        ratio_a = ratio_a[mask]
    return ratio_a - 1