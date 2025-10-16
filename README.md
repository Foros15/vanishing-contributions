# 🫧 Vanishing Contributions

[![arXiv](https://img.shields.io/badge/arXiv-2510.09696-b31b1b.svg)](https://arxiv.org/abs/2510.09696)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-EE4C2C?)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Contents <a id="contents"></a>
<details open>
<summary>Quick navigation</summary>

- [Abstract](#abstract)
- [Quick setup](#setup)
- [Train compressed models](#train-compressed-models)
- [Logging](#logging)
- [Citation](#citation)

</details>

## 🧠 Abstract <a id="abstract"></a>

> **Vanishing Contributions (VCON)** is a general approach for smoothly transitioning neural models into compressed form.
> Instead of abruptly switching to a compressed model, VCON blends the original and compressed networks during fine-tuning, gradually shifting their contributions.
> This smooth transition improves stability and mitigates accuracy loss, with consistent gains across computer vision and NLP benchmarks.

**Key Features:**
- 📉 Reduces memory, computation, and energy of the model
- 🛡️ Mitigates accuracy degradation
- 🔄 Works with pruning, quantization, low-rank, and more
- 📈 Typical gains: **3–20%** accuracy boost compared to compression + fine-tuning

**Working Mechanism**

$$
\bar g^t(x) = \beta^t f(x) + (1-\beta^t) g(x) 
$$
where $f$ is the original neural block, $g$ is its compressed form and $\bar g^t$ is the convex combination of the two. $\beta^t$ follows a linear schedule during training
$$
\beta^t = \max \left(1-\frac{t}{Q}, 0 \right)
$$
where $t$ is the current training step. When $t=Q$, $\beta^t = 0$ and training continues with the compressed form only.

---

## 🔧 Quick setup with conda <a id="setup"></a>

```bash
conda env create -f environment.yaml
conda activate vcon-env
```

### ⚡ Quick Smoke Test

In this repository, we prepared a VCON example training on ViT-T/16 (https://huggingface.co/WinKawaks/vit-tiny-patch16-224). You can easily modify the code to load ViT-S/16 or ViT-B/16 as in the experiments of our paper.

Verify your setup with a minimal run (vanilla training on Cifar-10). This test requires a maximum VRAM of ~5GB.

```bash
python train.py --model_name=vanilla-vit-tiny
```

## 🗜️ Train compressed models <a id="train-compressed-models"></a>

Compression is applied to all the layers in the models (both self-attention and multi-layer-perceptron blocks).

To launch layer-wise unstructured pruning (Remove 90\% of the interconnections) + fine-tuning:
```bash
python train.py --model_name=pruned-vit-tiny --prune_model
```

To launch STE-based Binary quantization-aware training:
```bash
python train.py --model_name=quantized-vit-tiny --quantize_model
```

To launch Low Rank Decomposition (Rank = 16) + fine-tuning:
```bash
python train.py --model_name=lrd-vit-tiny --lrd_model
```

### Train compressed models with Vanishing Contributions
To use VCON, you need to explicitly define the duration of the smooth transition into the compressed form (i.e., *_vcon_epochs)

```bash
python train.py --model_name=pruned-vcon-vit-tiny --prune_model --vcon_epochs=12
```

```bash
python train.py --model_name=quantized-vcon-vit-tiny --quantize_model --vcon_epochs=12
```

```bash
python train.py --model_name=lrd-vcon-vit-tiny --lrd_model --vcon_epochs=12
```

Look at vcon/config.py for more configuration options!

---

## 📂 Logging <a id="logging"></a>

You can inspect the training trends by launching tensorboard

```bash
tensorboard --logdir=models
```

You can also inspect some available pre-trained models

```bash
tensorboard --logdir=models_pretrained
```


## 📖 Citation <a id="citation"></a>

If you use our method, please cite:

```bibtex
@misc{nikiforosVanishingContributionsUnified2025,
  title = {Vanishing {{Contributions}}: {{A Unified Approach}} to {{Smoothly Transition Neural Models}} into {{Compressed Form}}},
  shorttitle = {Vanishing {{Contributions}}},
  author = {Nikiforos, Lorenzo and Antoniadis, Charalampos and Prono, Luciano and Pareschi, Fabio and Rovatti, Riccardo and Setti, Gianluca},
  year = {2025},
  month = oct,
  publisher = {arXiv},
  doi = {10.48550/arXiv.2510.09696},
  archiveprefix = {arXiv},
}
```

Plain text:  
L. Nikiforos, C. Antoniadis, L. Prono, F. Pareschi, R. Rovatti, and G. Setti,  
*"Vanishing Contributions: A Unified Approach to Smoothly Transition Neural Models into Compressed Form"*,  
October 9th, 2025, arXiv:2510.09696. doi: 10.48550/arXiv.2510.09696.

---