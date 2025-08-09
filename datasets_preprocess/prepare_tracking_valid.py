import numpy as np
from PIL import Image
import io
import argparse
import glob
import os
import tqdm
import json

def remove_folder(dir):
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))


def get_mask(mask, num_queries):
    indices = np.where(mask)[0]
    indices = np.random.choice(indices, num_queries, replace=False)
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[indices] = True
    return new_mask

def get_seq_data(input_path, output_path_base, seqs, min_num_queries=200):

    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
    image = Image.open(io.BytesIO(images_jpeg_bytes[0])).convert('RGB')
    W, H = image.size[0], image.size[1]
    fx, fy, cx, cy = intrinsics_params
    num_queries = queries_xyt.shape[0]

    max_time = len(images_jpeg_bytes)

    # 得到queries (t, n, 3)
    x = (tracks_xyz[:, :, 0] / tracks_xyz[:, :, 2]) * fx + cx
    y = (tracks_xyz[:, :, 1] / tracks_xyz[:, :, 2]) * fy + cy
    t = np.arange(max_time).reshape(-1, 1).repeat(num_queries, axis=1)
    queries = np.stack([x, y, t], axis=-1)
    masks = visibles & (x > 0) & (x < W) & (y > 0) & (y < H)
    num_valid = np.sum(masks, axis=1)

    for seq in seqs:
        start_frame = seq[0]
        mask = masks[start_frame]
        # print('num of valid queries is ', num_valid[start_frame],  '  ', np.sum(mask))
        if np.sum(mask) <= min_num_queries:
            continue
        mask = get_mask(mask, min_num_queries)    
        
        output_visibles = visibles[seq][:, mask]  # (2, n')
        output_tracks_xyz = tracks_xyz[seq][:, mask] #(2, n', 3)
        output_queries_xy = queries[start_frame, mask, :2] # (n', 2)
        output_queries_t = np.zeros_like(output_queries_xy[..., 0]).reshape(-1, 1) #(n', 1)
        output_queries = np.concatenate([output_queries_xy, output_queries_t], axis=1) #(n', 3)
        output_images_jpeg_bytes = images_jpeg_bytes[seq]
        
        in_npz_sequence = {
            'images_jpeg_bytes': output_images_jpeg_bytes,
            'queries_xyt': output_queries,
            'tracks_XYZ': output_tracks_xyz,
            'visibility': output_visibles,
            'fx_fy_cx_cy': intrinsics_params,
        }
        output_path = output_path_base + '_{}.npz'.format(start_frame)
        np.savez(output_path, **in_npz_sequence)



def get_seq_data_folder(input_path, output_path, seq_len, stride, size, clean):

    if os.path.exists(output_path) == False:
        os.makedirs(output_path, exist_ok=True)
    else:
        if clean:
            remove_folder(output_path)

    files = glob.glob(os.path.join(input_path, '**', '*.npz'), recursive=True)
    files = [os.path.basename(f) for f in files]

    for file in tqdm.tqdm(files):
        get_seq_data(os.path.join(input_path, file), os.path.join(output_path, file), seq_len, stride, size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='pair')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    
    data = json.load(open(args.config_path, 'r'))

    for dataset in data.keys():
        for config in data[dataset].keys():
            print(f'Processing dataset: {dataset}, config: {config}')
            output_dir = os.path.join(args.output_path, dataset + '_seq_' + config)
            if args.clean:
                remove_folder(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            for file_name in data[dataset][config].keys():
                input_path = os.path.join(args.input_path, dataset, file_name + '.npz')
                start_frames = data[dataset][config][file_name]
                output_path = os.path.join(output_dir, file_name)
                seq_len = int(config.split('_')[0])
                stride = int(config.split('_')[1])
                start_frames.sort()
                all_seqs = set()              
                for start_frame in start_frames:
                    indices = [start_frame + i * stride for i in range(seq_len)]
                    all_seqs.add(tuple(indices))
                seqs = [list(seq) for seq in all_seqs]
                get_seq_data(input_path, output_path, seqs)
