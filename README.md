<div align="center">
<h1>POMATO: Marrying Pointmap Matching with Temporal Motions
for Dynamic 3D Reconstruction</h1>

<p align="center">
<a href="https://arxiv.org/abs/2504.05692"><img src="https://img.shields.io/badge/ArXiv-2504.05692-%23840707.svg" alt="ArXiv"></a>
</p>

Songyan Zhang<sup>1*</sup>, Yongtao Ge<sup>2,3*</sup>, Jinyuan Tian<sup>2*</sup>, Hao Chen<sup>2†</sup>, Chen Lv<sup>1</sup>, Chunhua Shen<sup>2</sup>

<sup>1</sup>Nanyang Technology University, Singapore; <sup>2</sup>Zhejiang University, China; <sup>3</sup>The University of Adelaide, Australia

*Equal Contributions, †Corresponding Author
<br><br><image src="./assets/teaser.png"/>
</div>

We present **POMATO** , a model that enables 3D reconstruction from an arbitrary dynamic video. Without relying on external modules, POMATO can
directly perform 3D reconstruction along with temporal 3D point tracking and dynamic mask estimation.

# Code will come soon!

## 🚀News

- ```[Apr 2025]``` Released [paper](https://arxiv.org/abs/2504.05692) and init the github repo.


## 🔨 TODO LIST

- [ ] Release the inference code and huggingface model.
- [ ] Release the visualization of 3D tracking.
- [ ] Release the training code.

## ✨Hightlights
🔥 We introduce a temporal motion module to facilitate the interactions of motion features along the temporal dimension.

<p align='center'>
    <image src="./assets/temporal_infer.png"/>
</p>

Inference pipelines for point tracking, video depth,
and multi-view reconstruction with temporal motion module. tk indicates the keyframe in the
sequence.

## 📌 Citation

If you find our POMATO is useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:

```bibtex
@article{zhang2025pomatomarryingpointmapmatching,
  title={POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction}, 
  author={Songyan Zhang and Yongtao Ge and Jinyuan Tian and Guangkai Xu and Hao Chen and Chen Lv and Chunhua Shen},
  journal={arXiv preprint arXiv:2504.05692},
  year={2025},
}
```
