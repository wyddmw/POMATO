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
we simply put the core parts of our model and integrate it into MonST3R's framework for inference.
'''

class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, freeze='none', **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)
        self.set_freeze(freeze)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

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
        self.downstream_head1 = head_factory(head_type='dpt', output_mode='pts3d', net=self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type='dpt', output_mode='pts3d', net=self, has_conf=bool(conf_mode))
        self.downstream_head3 = head_factory(head_type='dpt', output_mode='pts3d', net=self, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        self.head3 = transpose_to_landscape(self.downstream_head3, activate=landscape_only)

    def forward(self, view1, view2=None):
        
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)            
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
            res3 = self._downstream_head(3, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        res2['pts3d_dynamic'] = res3.pop('pts3d')
        res2['conf_dynamic'] = res3.pop('conf')
        return res1, res2