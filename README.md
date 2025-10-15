# 🚀 Vanishing Contributions

[![arXiv](https://img.shields.io/badge/arXiv-2510.09696-b31b1b.svg)](https://arxiv.org/abs/2510.09696)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Abstract

> **Vanishing Contributions (VCON)** is a general approach for smoothly transitioning neural models into compressed form.  
> Instead of abruptly switching to a compressed model, VCON blends the original and compressed networks during fine-tuning, gradually shifting their contributions.  
> This smooth transition improves stability and mitigates accuracy loss, with consistent gains across computer vision and NLP benchmarks.

**Key Features:**
- 📉 Reduces memory, computation, and energy of the model
- 🛡️ Mitigates accuracy degradation
- 🔄 Works with pruning, quantization, low-rank, and more
- 📈 Typical gains: **3–20%** accuracy boost compared to compression + fine-tuning

---

## ⚡ Quick Smoke Test

Verify your setup with a minimal run (vanilla training on Cifar-10):

```bash
python train.py --model_name=vanilla-vit-tiny
```

## Train compressed models

### Layer-wise unstructured pruning + fine-tuning
```bash
python train.py --model_name=pruned-vit-tiny --prune_model
```

### STE-based Binary quantization-aware training
```bash
python train.py --model_name=quantized-vit-tiny --quantize_model
```

### Low Rank Decomposition + fine-tuning
```bash
python train.py --model_name=lrd-vit-tiny --lrd_model
```

## Train compressed models with Vanishing Contributions
You need to explicitly define the duration of the smooth transition into the compressed form (i.e., *_vcon_epochs)

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

## 📂 Logging

You can inspect the training trends by launching tensorboard

```bash
tensorboard --logdir=models
```


## 📖 Citation

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