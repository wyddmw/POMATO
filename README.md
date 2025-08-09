<div align="center">
<h1>POMATO: Marrying Pointmap Matching with Temporal Motions
for Dynamic 3D Reconstruction</h1>

<p align="center">
<a href="https://arxiv.org/abs/2504.05692"><img src="https://img.shields.io/badge/ArXiv-2504.05692-%23840707.svg" alt="ArXiv"></a>
</p>

[![🤗 HuggingFace models](https://img.shields.io/badge/HuggingFace🤗-Models-orange)](https://huggingface.co/wyddmw/WiseAD)


Songyan Zhang<sup>1*</sup>, Yongtao Ge<sup>2,3*</sup>, Jinyuan Tian<sup>2*</sup>, Hao Chen<sup>2†</sup>, Chen Lv<sup>1</sup>, Chunhua Shen<sup>2</sup>

<sup>1</sup>Nanyang Technological University, Singapore; <sup>2</sup>Zhejiang University, China; <sup>3</sup>The University of Adelaide, Australia

*Equal Contributions, †Corresponding Author
<br><br><image src="./assets/teaser.png"/>
</div>

We present **POMATO** , a model that enables 3D reconstruction from an arbitrary dynamic video. Without relying on external modules, POMATO can directly perform 3D reconstruction along with temporal 3D point tracking and dynamic mask estimation.

## 📰News

- ```[Apr 2025]``` Released [paper](https://arxiv.org/abs/2504.05692) and init the github repo.
- ```[June 2025]``` POMATO was accepted to ICCV 2025🎉🎉🎉!
- ```[July 2025]``` Released [pre-trained models](https://huggingface.co/wyddmw/POMATO) (pairwise and temporal versions) on huggingface and related inference code.

## 🔨 TODO LIST

- [x] Release the inference code and huggingface model.
- [x] Release the pose evaluation code.
- [x] Release the visualization and evaluation of 3D tracking.
- [ ] Release the video depth evaluation.
- [ ] Release the training code.

## 🚀 Getting Started

### Installation

1. **Clone the repository and set up the environment:**
```bash
git clone https://github.com/wyddmw/POMA_eval.git
cd POMATO
```

2. **Install dependencies:**
Follow MonST3R to build the conda environment.
```bash
conda create -n pomato python=3.11 cmake=3.14.0
conda activate pomato 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - training
# - evaluation on camera pose
# - dataset preparation
pip install -r requirements_optional.txt
```
2. **Optional, install 4d visualization tool, viser.**
```bash
pip install -e viser
```

2. **Optional, compile the cuda kernels for RoPE (as in CroCo v2).**
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Download Model Weights

Download the pre-trained POMATO model weights and place them under pretrained_models/.
- **POMATO Pairwise Model & POMATO Temporal Model**:  Available [Here](https://huggingface.co/wyddmw/POMATO).

### Quick Demo
Play with the demo data for 3D reconstruction.
```bash
bash demo.sh
```
The estimated depth and dynamic masks are saved in ./recon_results. Check the visualization at http://0.0.0.0:8080.

### Fast 3D Reconstruction Inference
Use our pre-trained temporal models for temporal enhanced 3D reconstruction in a feed-forward manner. If the input video sequence is less than target 12 frames, the last frame will be repeated for padding.
```python
python inference_scripts/fast_recon_temp.py --model pretrained_models/SPECIFIC_PRETRAINED_MODEL --image_folder YOUR/IMAGE/FOLDER
# an inference example:
# python inference_scripts/fast_recon_temp.py --model pretrained_models/POMATO_temp_6frames.pth --image_folder asssets/davis/
```

### Pose Evaluation
Following [download instruction](https://github.com/Junyi42/monst3r/blob/main/data/prepare_training.md) to prepare the datasets first. Then construct the sampled data on Bonn and TUM datasets.
```python
python datasets_preprocess/prepare_bonn.py  # sample with interval 5.
```
Run the evaluation script:
```bash
bash eval_scrips/eval_pose.sh
```

### Tracking
Prepare the tracking validation data.

For `adt` and `pstudio`, due to the restriction of the original data lisence, we can't provided the processed data. However, we provide the script to process the original data.

First, follow the guidance of [TAPVid-3D](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d). 
Download the minival dataset to `data/`

For example, after creating correct environment, run
```
python3 -m tapnet.tapvid3d.annotation_generation.generate_adt \
--output_dir [path to tapvid datasets] \
--split=minival \
```

```
python3 -m tapnet.tapvid3d.annotation_generation.generate_pstudio \
--output_dir [path to tapvid datasets] \
--split=minival \
```

Then prepare the validation data:
```
python datasets_preprocess/prepare_tracking_valid.py \
--input_path data/tapvid_datasets \
--output_path data/tracking_eval_data \
--config_path datasets_preprocess/configs/tracking_valid.json \
```

For `PointOdyssey`, we provide the link to download the processed data: [POMATO_Tracking](https://huggingface.co/datasets/xiaochui/POMATO_Tracking/tree/main).
```
huggingface-cli download xiaochui/POMATO_Tracking po_seq.zip  --local-dir ./data/tracking_eval_data
```

After downloading the data, unzip it to `data/tracking_eval_data`
```
cd data/tracking_eval_data 
unzip po_seq.zip
```

Modify the input path, output path and weight path before running the script if needed. The outputs will be saved as `xxx.npz` files.(For more details, please refer to inference_scripts/tracking.sh)
```
bash inference_scripts/tracking.sh
```

**Please note that during the data generation process for ADT and PStudio, the sampled data and their start time configurations are consistent with those in the original paper. However, since the query points are sampled randomly, the results may differ slightly from those reported in the paper.**

We provide a random result to compare it with the result in paper.
| Result         | adt-12 | adt-24 | pstudio-12 | pstudio-24 | po-12 | po-24 |
| -------------- | ------ | ------ | ---------- | ---------- | ----- | ----- |
| In Paper       | 31.57  | 28.22   | 24.59     | 19.79     | 33.20 | 33.58 |
| Random Result  | 31.35  | 28.11   | 24.56      | 19.62     | 33.20 | 33.58 |

## 📌 Citation

If you find our POMATO is useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:

```bibtex
@article{zhang2025pomato,
  title={POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction},
  author={Zhang, Songyan and Ge, Yongtao and Tian, Jinyuan and Xu, Guangkai and Chen, Hao and Lv, Chen and Shen, Chunhua},
  journal={arXiv preprint arXiv:2504.05692},
  year={2025}
}
```

## 🙏 Acknowledgements
Our code is based on [MonST3R](https://github.com/Junyi42/monst3r), [DUSt3R](https://github.com/naver/dust3r), and [MASt3R](https://github.com/naver/mast3r). We appreciate the authors for their excellent works! We also thank the authors of [GCD](https://github.com/basilevh/gcd) for their help on the ParallelDomain-4D dataset.