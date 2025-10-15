import os
import torch
from transformers import ViTForImageClassification


def build_model(cfg, num_classes):
    model = ViTForImageClassification.from_pretrained(
        cfg.vit_model,
        ignore_mismatched_sizes=True,
        num_labels=num_classes,
    )
    if cfg.load_model:
        if os.path.isdir(cfg.load_model):
            state = torch.load(os.path.join(cfg.load_model, 'model.pt'), weights_only=True)
            model.load_state_dict(state)
            print('Loaded model from', os.path.join(cfg.load_model, 'model.pt'))
        else:
            state = torch.load(cfg.load_model, weights_only=True)
            model.load_state_dict(state)
            print('Loaded model from', cfg.load_model)
    return model
