import torch
from dust3r.utils.tracking import get_images, get_image_pairs, solve_scale_shift, interpolate_overlap, interpolate_at_xy, check_dir
from dust3r.inference import inference
import numpy as np
import os
import glob
import tqdm

def track_single_file(model, input_path, output_path, device, window, overlap, model_type='temporal'):
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        origin_intrinsics_params = in_npz['fx_fy_cx_cy']

    if model_type == 'POMATO_temp_12frames':
        num_frames = len(images_jpeg_bytes)
        true_window = window - 1 
        step = true_window - overlap
        prediction_tracks_xyz = np.zeros_like(tracks_xyz)
        keyframe_index = 0
        cnt = 0 
        for start in range(1, num_frames, step):
            indices = [keyframe_index] + list(range(start, min(start + true_window, num_frames)))
            true_window_len = len(indices)
            if len(indices) < window:
                indices += indices[-1:] * (window - len(indices))
            images, origin_shape, current_shape, t = get_images(images_jpeg_bytes, indices, device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    output = model(images, inference=True)
            pts = output['pts3d_matching']
            x, y = queries_xyt[..., 0] / origin_shape[0] * current_shape[0], queries_xyt[..., 1] / origin_shape[1] * current_shape[1]
            prediction_tracks_xyz_window = interpolate_at_xy(pts, x, y).permute(0, 2, 1).cpu().numpy()
            if start == 1:
                prediction_tracks_xyz[indices, ...] = prediction_tracks_xyz_window
            else:
                prediction_tracks_xyz[keyframe_index] += prediction_tracks_xyz_window[0]
                overlap_indices = indices[1: 1 + overlap]
                if len(overlap_indices) > 0:
                    prediction_tracks_xyz[overlap_indices] = interpolate_overlap(prediction_tracks_xyz[overlap_indices], prediction_tracks_xyz_window[1: 1 + overlap])
                other_indices = indices[1 + overlap:true_window_len]
                prediction_tracks_xyz[other_indices] = prediction_tracks_xyz_window[1 + overlap:true_window_len]
            
            cnt += 1

        prediction_tracks_xyz[0] /= cnt 
    elif model_type == 'POMATO_pairwise':
        query_time = queries_xyt[:, 2] # from indx to time
        query_idx = list(set(int(idx) for idx in queries_xyt[:, 2])) # possible query time
        
        image_pairs, origin_shape, current_shape, t = get_image_pairs(images_jpeg_bytes, query_idx)
        
        # turn to the data for TAPVid3D
        prediction_tracks_xyz = np.zeros_like(tracks_xyz)
        prediction_visibles = np.zeros_like(visibles)

        for current_query_idx in range(len(query_idx)):
            current_query_time = query_idx[current_query_idx]
            output = inference(list(image_pairs[:, current_query_idx]), model, device, batch_size=24, verbose=False)

            # at this stage, you have the raw dust3r predictions
            pred1 = output['pred1']
            conf = pred1['conf'].view(t, current_shape[1], current_shape[0])
            
            pts = output['pred2']['pts3d_dynamic'].view(t, current_shape[1], current_shape[0], 3) # t * #query_frames, h, w, 3

            # x,y是time frame上的坐标，而不是任意帧上的坐标，得找到对应的点才行
            mask = query_time == current_query_time
            x, y = queries_xyt[mask, 0] / origin_shape[0] * current_shape[0], queries_xyt[mask, 1] / origin_shape[1] * current_shape[1]
            prediction_tracks_xyz[:, mask] = interpolate_at_xy(pts, x, y).permute(0, 2, 1)
            prediction_visibles[:, mask] = interpolate_at_xy(conf.unsqueeze(-1), x, y).permute(0, 2, 1).squeeze(-1)


    # scale shift normalization
    scale, shift = solve_scale_shift(prediction_tracks_xyz, tracks_xyz, visibles)
    prediction_tracks_xyz = scale * prediction_tracks_xyz + shift

    output = {
        'tracks_XYZ': prediction_tracks_xyz,
        'visibility': visibles,
        'fx_fy_cx_cy': origin_intrinsics_params,
        'images_jpeg_bytes': images_jpeg_bytes,
        'queries_xyt': queries_xyt,
    }
    np.savez(output_path, **output)



def tracking(model, args):
    datasets = args.dataset.split(',')
    for dataset in datasets:
        check_dir(os.path.join(args.output_path, dataset), args.clean) 
        print('Start tracking dataset: ', dataset)
        files = glob.glob(os.path.join(args.input_path, dataset, '**', '*'), recursive=True)
        files = [os.path.basename(f) for f in files]
        for file in tqdm.tqdm(files):
            track_single_file(model, os.path.join(args.input_path, dataset, file), os.path.join(args.output_path, dataset, file), args.device, args.window, args.overlap, args.model_type)
    