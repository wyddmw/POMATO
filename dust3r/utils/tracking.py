import torch
import torch.nn.functional as F

import numpy as np
import io
from PIL import Image
import os

import torchvision.transforms as tvf

def check_dir(path, delete=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if delete:
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_images(images_jpeg_bytes, indices, device, size=512, save_imgs=False, factor = 16):
    imgs = []
    idx = 0
    origin_shape = ()
    current_shape = ()
    for index in indices:
        image_jpeg_bytes = images_jpeg_bytes[index]
        idx += 1
        img = Image.open(io.BytesIO(image_jpeg_bytes)).convert('RGB')
        W1, H1 = img.size
        
        # 找到最长边
        max_edge = max(W1, H1)
        
        # 计算缩放比例
        scale = size / max_edge
        
        # 根据比例缩放两边
        W2 = int(W1 * scale)
        H2 = int(H1 * scale)
        
        # 将宽高调整为factor的倍数
        W2 = (W2 ) // factor * factor
        H2 = (H2 ) // factor * factor
        
        img = img.resize((W2, H2))

        imgs.append(ImgNorm(img)[None])
        
        if save_imgs:
            img.save('output/image' + str(idx) + '.png')
        
        if(idx == 1):
            origin_shape = (W1, H1)
            current_shape = (W2, H2)
    t = len(imgs)
    imgs = torch.concat(imgs)[None].to(device).to(torch.float16)
    return {'img': imgs}, origin_shape, current_shape, t

def get_image_pairs(images_jpeg_bytes, query_idx, size=512, save_imgs=False, factor = 16):
    imgs = []
    img_pairs = np.empty((len(images_jpeg_bytes), len(query_idx)), dtype=tuple)
    idx = 0
    origin_shape = ()
    current_shape = ()
    for image_jpeg_bytes in images_jpeg_bytes:
        idx += 1
        img = Image.open(io.BytesIO(image_jpeg_bytes)).convert('RGB')
        W1, H1 = img.size
        
        # 找到最长边
        max_edge = max(W1, H1)
        
        # 计算缩放比例
        scale = size / max_edge
        
        # 根据比例缩放两边
        W2 = int(W1 * scale)
        H2 = int(H1 * scale)
        
        # 将宽高调整为factor的倍数
        W2 = (W2 ) // factor * factor
        H2 = (H2 ) // factor * factor
        
        img = img.resize((W2, H2))

        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        
        if save_imgs:
            img.save('output/image' + str(idx) + '.png')
        
        if(idx == 1):
            origin_shape = (W1, H1)
            current_shape = (W2, H2)
    
    # 每一个pair --- (任意时刻的frame, 需要track的frame的indx)
    for i in range(len(imgs)):
        for j in range(len(query_idx)):
            img_pairs[i][j] = tuple([imgs[i], imgs[j]])
    
    return img_pairs, origin_shape, current_shape, len(imgs)

def interpolate_at_xy(pts, x, y):
    """
    在 (x, y) 位置对 shape 为 (n, h, w, 3) 的张量进行插值
    
    参数:
    pts: 形状为 (n, h, w, 3) 的张量
    x, y: 单个插值点，分别是 x 和 y 坐标
    
    返回:
    形状为 (n, 3) 的插值结果
    """
    n, h, w, c = pts.shape
    
    # 将坐标归一化到 [-1, 1] 区间
    norm_x = 2.0 * x / (w - 1) - 1.0
    norm_y = 2.0 * y / (h - 1) - 1.0
    
    # 创建归一化的坐标网格, shape为 (1, 1, 2)
    norm_coords = torch.stack([torch.tensor(norm_x), torch.tensor(norm_y)], dim=-1).to(pts.device).to(torch.float32)
    
    # 调整 pts 形状为 (n, 3, h, w) 以适配 grid_sample
    pts = pts.permute(0, 3, 1, 2)  # 转换为 (n, 3, h, w)
    
    # norm_coords 扩展为 (n, 1, #query_xy, 2)
    norm_coords = norm_coords.expand(n, 1, -1, -1)
    # 使用 grid_sample 进行插值, 插值结果 shape 为 (n, 3, 1, 1)
    interpolated_vals = F.grid_sample(pts, norm_coords, mode='bilinear', align_corners=True)
    
    # 去掉多余的维度，返回形状为 (n, 3) 的结果
    return interpolated_vals.squeeze(-2)

def calculate_intrinsic(pts, conf, x_coords, y_coords, cx, cy, percentile):
    mask = conf > torch.quantile(conf.flatten(), percentile)
    fx = (x_coords - cx) / (pts[..., 0] / (pts[..., 2] + 1e-8))
    fy = (y_coords - cy) / (pts[..., 1] / (pts[..., 2] + 1e-8))
    mask_fx, mask_fy = fx[mask], fy[mask]
    result_fx, result_fy = mask_fx.nanmedian().float(), mask_fy.nanmedian().float()
    return result_fx, result_fy

def solve_scale_shift(pcs1, pcs2, mask):
   # pts (h, w, 3)  or (n, 3)
    # h, w, _ = pcs1.shape
    s, t = np.zeros(3), np.zeros(3)
    mask = mask.reshape(-1)
    a, b = pcs1.reshape(-1, 3)[mask], pcs2.reshape(-1, 3)[mask]
    n = a.shape[0]

    for i in range(3):
        A_aug = np.column_stack((a[:, i], np.ones(n)))
        b_col = b[:, i]
        x, _, _, _ = np.linalg.lstsq(A_aug, b_col, rcond=None)
        s[i], t[i] = x[0], x[1]
    
    return s, t

def image_bytes_to_tensor(image_bytes):
    # (h, w, 3)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = ImgNorm(image)
    return image

def save_pts(save_path, point_clouds, frames, intrs, mask):
    '''
    torch.tensor
    point_clouds: (n, (h, w, 3)) [pcs1, pcs2, ...]
    frame: (n, (h, w, 3)) [frame1, frame2, ...]
    mask: (h, w)
    '''
    if mask == None:
        mask = torch.ones_like(point_clouds[0][..., 0], dtype=torch.bool)
    n = len(point_clouds)
    save_data = {
        'point_clouds': torch.stack(point_clouds, dim=0).cpu().numpy(),
        'frames': torch.stack(frames, dim=0).cpu().numpy(),
        'intrs': torch.tensor(intrs).unsqueeze(0).repeat(n, 1).cpu().numpy(),
        'extrs': None,
        'masks': mask.squeeze().unsqueeze(0).repeat(n, 1, 1).cpu().numpy(),
    }
    np.save(save_path, save_data)

def interpolate_overlap(A, B):
    # 确保A和B的shape相同
    assert A.shape == B.shape, "A and B must have the same shape"
    
    # 计算插值系数
    a_dim = A.shape[0]
    t = np.linspace(0.2, 0.8, a_dim)  # 从0到1，a_dim个插值点
    
    # 利用广播机制进行向量化计算
    result = (1 - t[:, None, None]) * A + t[:, None, None] * B
    
    return result