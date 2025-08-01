import cv2
import PIL
import numpy as np
import argparse
import os
import glob

from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from dust3r.utils.image import rgb
from dust3r.viz import SceneViz, auto_cam_size
import torch
import torch.nn.functional as F
from dust3r.POMATO_temp import AsymmetricMASt3R


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def load_images_from_folder(folder_path, size, device, target_frames=12):
    """
    Load images from a folder, sort them, and pad with last frame if needed.
    
    Args:
        folder_path: Path to folder containing images
        size: Image size for resizing
        device: Device to load tensors on
        target_frames: Target number of frames (default 12)
    
    Returns:
        Dictionary with 'img' tensor and temporal_length
    """
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Get all image files from the folder
    img_paths = []
    for ext in image_extensions:
        img_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        img_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not img_paths:
        raise ValueError(f"No images found in folder: {folder_path}")
    
    # Sort the image paths
    img_paths.sort()
    print(f"Found {len(img_paths)} images in {folder_path}")
    
    # If we have fewer than target_frames, pad with the last frame
    if len(img_paths) < target_frames:
        last_frame = img_paths[-1]
        padding_needed = target_frames - len(img_paths)
        print(f"Padding with last frame {padding_needed} times to reach {target_frames} frames")
        img_paths.extend([last_frame] * padding_needed)
    elif len(img_paths) > target_frames:
        # If we have more than target_frames, take the first target_frames
        print(f"Using first {target_frames} frames out of {len(img_paths)} available")
        img_paths = img_paths[:target_frames]
    
    return load_images(img_paths, size, device)

def load_images(img_dirs, size, device):
    imgs = []
    for path in img_dirs:
        img = exif_transpose(PIL.Image.open(path)).convert('RGB')
        img = _resize_pil_image(img, size)

        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        imgs.append(ImgNorm(img)[None])

    imgs = torch.concat(imgs)[None].to(device).to(torch.float16)
    return {'img': imgs}

def parse_args():
    parser = argparse.ArgumentParser(description='Fast 3D reconstruction from temporal images')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to folder containing images')
    parser.add_argument('--model', type=str, 
                        default='./pretrained_models/POMATO_temp_6frames.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--size', type=int, default=512,
                        help='Image size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--conf_threshold', type=float, default=1.3,
                        help='Confidence threshold for filtering points')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Validate folder exists
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Folder does not exist: {args.image_folder}")
    if not os.path.isdir(args.image_folder):
        raise ValueError(f"Path is not a directory: {args.image_folder}")
    device = args.device

    model_name = args.model
    # Check if model exists
    if not os.path.exists(model_name):
        raise ValueError(f"Model checkpoint does not exist: {model_name}")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)    
    temp_length = model.temporal_length
    # Load images from the specified folder
    images = load_images_from_folder(args.image_folder, args.size, device, temp_length)
    model.eval()
    mode = 'fast_recon'
    keyframe_index = temp_length - 1
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            output = model(images, inference=True, mode=mode)
            pred3 = output['recon_pts3d'].data.cpu().numpy()
            conf_pred1 = output['recon_conf'].data.cpu().numpy() > args.conf_threshold
            conf_pred1 = conf_pred1.reshape(-1)
            pred3 = pred3.reshape(-1, 3)
            img = images['img'].data.cpu().numpy()
            color = rgb(img)
            color = color.reshape(-1, 3)
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
            torch.cuda.empty_cache()

    print('Starting 3D visualization...')
    # Create SceneViz instance
    viz = SceneViz()
    
    # Apply confidence mask to filter low-confidence points
    valid_pred3 = pred3[conf_pred1]
    valid_colors = color[conf_pred1]
    # Add the reconstructed 3D points with RGB colors
    viz.add_pointcloud(valid_pred3, valid_colors)
    viz.show()
    print("Visualization complete!")

        