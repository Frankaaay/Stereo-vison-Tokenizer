# OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation

> **Stereo-only development branch.** The public tokenizer in this branch is
> no longer the upstream image/video VQGAN. It accepts structured
> `[B,3,2,3,4,256,256]` stereo samples, produces a raw
> `[B,3,48,1,16,16]` VAE latent, and decodes left-view RGB plus disparity.
> Legacy image-mode and VQ codebook paths are not supported. The upstream
> model-zoo checkpoints and LM/DiT/Latte entrypoints below are retained as
> historical repository context and are not strict-load compatible with this
> Stereo model.

## Stereo OmniTokenizer

The implementation lives in the original repository path:

- `OmniTokenizer/omnitokenizer.py`: shared per-frame Spatial Encoder,
  StereoFusion, `4→1` temporal projection, VAE posterior, original
  one-slot temporal/spatial Decoder, RGB/disparity heads, and training losses.
- `OmniTokenizer/modules/stereo_*.py`: fusion, geometry, and masked loss
  primitives.
- `OmniTokenizer/data.py`: Manifest v3 loader for independent RGB and
  FoundationStereo GT caches.
- `scripts/data/build_stereo_rgb_cache.py`: independent RGB-cache builder and
  Manifest v3 finalizer.
- `vqgan_train.py`, `vqgan_eval.py`, and `scripts/recons/train.sh`: Stereo-only
  training/evaluation entrypoints.

The tokenizer intentionally does not implement downstream DiT
patchify/unpatchify. See `doc/Stereo Tokenizer Plan.md` for the frozen tensor,
data, supervision, and validation contracts. H200 smoke/overfit execution is a
separate gated step; the repository does not contain datasets, caches,
checkpoints, or run outputs.

Official pytorch implementation of the following paper:
<p align="left"> 
<a href="https://arxiv.org/abs/2406.09399">OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation</a>.
<br>
<br>
<a href="https://www.wangjunke.info/">Junke Wang</a><sup>1,2</sup>, <a href="https://enjoyyi.github.io/">Yi Jiang</a><sup>3</sup>, <a href="https://shallowyuan.github.io/">Zehuan Yuan</a><sup>3</sup>, <a href="./">Binyue Peng</a><sup>3</sup>, <a href="https://zxwu.azurewebsites.net/">Zuxuan Wu</a><sup>1,2</sup>, <a href="https://fvl.fudan.edu.cn/">Yu-Gang Jiang</a><sup>1,2</sup>
<br>
<sup>1</sup>Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University <br>
<sup>2</sup>Shanghai Collaborative Innovation Center of Intelligent Visual Computing, <sup>3</sup>Bytedance Inc.
</p>

<p align="left">
    <img src=assets/network.png width="852" height="284" />
</p>


We introduce OmniTokenizer, a joint image-video tokenizer which features the following properties:
- 🚀 **One model** and **one weight** for joint image and video tokenization;
- 🥇 **State-of-the-art reconstruction performance** on both image and video datasets;
- ⚡ High adaptability to **high resolution** and **long** video inputs;
- 🔥 Equipped with it, both **language model** and **diffusion model** could achieve competitive visual generation results.

Please refer to our [project page](https://www.wangjunke.info/OmniTokenizer/) for the reconstruction and generation results by OmniTokenizer.

## Setup

Please setup the environment using the following commands:

```
pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

Then download the datasets from the official websites. You can download the [annotation.zip](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/annotations.zip) processed by us and put them under ```./annotations```.

## Upstream Model Zoo for VQVAE and VAE (incompatible with Stereo-only class)

We release both VQVAE and VAE version of OmniTokenizer, that are pretrained on a wide range of image and video datasets:

 |  Type | Training Data  | FID | FVD | ckpt | 
 | ---------- | ---------- | ---------- | ----------- | ----------- | 
 | VQVAE | ImageNet | 1.28[^1] | - | [imagenet_only.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_only.ckpt) |
 | VQVAE | CelebAHQ | 1.85 | - | [celebahq.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq.ckpt) | 
 | VQVAE | FFHQ |2.58 | - | [ffhq.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ffhq.ckpt) | 
 | VQVAE | ImageNet + UCF | 1.11 | 42.35 | [imagenet_ucf.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_ucf.ckpt) | 
 | VQVAE | ImageNet + K600 | 1.23 | 25.97 | [imagenet_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_k600.ckpt) | 
 | VQVAE | ImageNet + MiT | 1.26 | 19.87 | [imagenet_mit.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_mit.ckpt) | 
 | VQVAE | ImageNet + Sthv2 | 1.21 | 20.30 | [imagenet_sthv2.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_sthv2.ckpt) | 
 | VQVAE | CelebAHQ + UCF | 1.93 | 45.59 | [celebahq_ucf.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq_ucf.ckpt) | 
 | VQVAE | CelebAHQ + K600 | 1.82 | 89.13 | [celebahq_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq_k600.ckpt) | 
 | VQVAE | FFHQ + UCF | 1.91 | 57.93 | [ffhq_ucf.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ffhq_ucf.ckpt) | 
 | VQVAE | FFHQ + K600 | 2.69 | 87.58 | [ffhq_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ffhq_k600.ckpt) | 
 | VAE | ImageNet + UCF | 0.69 | 23.44 | [imagenet_ucf_vae.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_ucf_vae.ckpt) | 
 | VAE | ImageNet + K600 | 0.78 | 13.02 | [imagenet_k600_vae.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_k600_vae.ckpt) |

[^1] We train this model w/o *scaled_dot_product_attention*, please comment line 446-460 in ```OmniTokenizer/modules/attention.py``` to reproduce this result.


These links and reported metrics describe the upstream implementation. They
must not be used as pretrained weights for this branch: Stereo training starts
from scratch and evaluation uses strict checkpoint loading through
`vqgan_eval.py`.

## Stereo Tokenizer Training

`scripts/recons/train.sh` is the canonical recipe template. It requires the
Manifest v3/cache paths and all not-yet-calibrated loss, batch, warmup, and
step-budget values as environment variables. GAN is explicitly disabled in
the first smoke/overfit recipe. A validation Manifest is optional for the
engineering pilot; when supplied for formal data, the complete validation
split runs once at each epoch end.


## LM-based Visual Synthesis

The upstream LM scripts below target discrete codebook tokens and are not an
entrypoint for the raw 48-channel Stereo VAE latent.

Please refer to ```scripts/lm_train``` and ```scripts/lm_gen``` for the training and evaluation of language model. We provide the checkpoints for ImageNet[[imagenet_class_lm.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_class_lm.ckpt)], UCF [[ucf_class_lm.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ucf_class_lm.ckpt)], and Kinetics-600 [[k600_fp_lm.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/k600_fp_lm.ckpt)]. 

## Diffusion-based Visual Synthesis

The upstream diffusion scripts below are historical. Downstream Stereo latent
normalization and DiT patchify/unpatchify belong to the consuming model and are
not implemented in this tokenizer.

We adopt [DiT](https://github.com/facebookresearch/DiT?tab=readme-ov-file) and [Latte](https://github.com/Vchitect/Latte) for diffusion-based visual generation. Please refer to [diffusion.md](Diffusion/README.md) for the training and evaluation instructions.

## Evaluation

Please refer to [evaluation.md](evaluation/README.md) for how to evaluate the reconstruction or generation results.

## Acknowledgments
Our code is partially built upon [VQGAN](https://github.com/CompVis/taming-transformers) and
[TATS](https://github.com/songweige/TATS). We also appreciate the wonderful tools provided by [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality).



## License

This project is licensed under the MIT license, as found in the LICENSE file.
