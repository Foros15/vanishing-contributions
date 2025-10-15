import os
import argparse
import configparser
from types import SimpleNamespace
from typing import Dict

from utils import combine_config, convert_string


def default_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg['System-setup'] = {
        'num_workers': 36,
        'seed': 42,
    }
    cfg['Model'] = {
        'model_name': 'vit-tiny',
        'store_folder': 'models',
        'load_model': '',
        'vit_model': 'WinKawaks/vit-tiny-patch16-224',
    }
    cfg['Task'] = {
        'dataset': 'cifar10',
        'dataset_root': '.',
    }
    cfg['Online RS-STE pruning'] = {
        'prune_model': False,
        'prune_ratio': 0.9,
        'prune_granularity': 'layer',
        'prune_lambda_w': 0.0,
        'prune_vcon_epochs': 0,
    }
    cfg['Online STE quantization'] = {
        'quantize_model': False,
        'quantize_vcon_epochs': 0,
    }
    cfg['Low Rank Decomposition'] = {
        'lrd_model': False,
        'lrd_rank': 16,
        'lrd_vcon_epochs': 0,
    }
    cfg['Batch and epochs'] = {
        'batch_size': 128,
        'batch_accumulation': 1,
        'starting_epoch': 0,
        'total_epochs': 60,
    }
    cfg['Learning rate scheduling'] = {
        'learning_rate': 1e-4,
        'lr_decay_factor': 1.0,
        'lr_patience': 10000,
        'lr_minimum': 1e-6,
        'lr_warmup_epochs': 1,
        'lr_warmup_start': 1e-5,
        'lr_cosine_decay': True,
    }
    cfg['Regularizations'] = {
        'weight_decay': 0,
        'clipping_global_norm': False,
        'wsatstd': '',
    }
    cfg['Dataugmentation'] = {
        'use_cutmix_mixup': True,
        'use_mixup_only': True,
        'mixup_alpha': 1.0,
        'use_autoaugment': False,
        'use_randaugment': True,
    }
    return cfg


def _is_bool_value(v) -> bool:
    try:
        return str(v).lower() in {"true", "false"}
    except Exception:
        return False


def build_arg_parser(cfg: configparser.ConfigParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='VCON training')

    # dynamic flags: support presence-only booleans using underscored names
    for section in cfg:
        for key in cfg[section]:
            val = cfg[section][key]
            is_bool = _is_bool_value(val)
            opt = f"--{key}"
            if is_bool:
                # presence sets True; allow explicit disable via --no_key
                parser.add_argument(opt, dest=key, action='store_true', default=None, help=f"Enable {section}:{key}")
                parser.add_argument(f"--no_{key}", dest=key, action='store_false', default=None, help=f"Disable {section}:{key}")
            else:
                parser.add_argument(opt, type=str, help=f"Override {section}:{key}")

    parser.add_argument('-c', '--config', type=str, default='', help='Path to config INI or a dir containing config.ini')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite config.ini if exists')
    parser.add_argument('-t', '--test', action='store_true', help='Run test only and exit')
    return parser


def load_and_merge_config(cfg: configparser.ConfigParser, args: argparse.Namespace) -> configparser.ConfigParser:
    # Load from file if provided
    if args.config:
        path = args.config
        if os.path.isdir(path):
            path = os.path.join(path, 'config.ini')
        if os.path.exists(path):
            loaded = configparser.ConfigParser()
            loaded.read(path)
            for section in loaded:
                for key in loaded[section]:
                    if section in cfg and key in cfg[section]:
                        val = loaded[section][key]
                        if val is not None:
                            cfg[section][key] = val
        else:
            print(f"Config file {args.config} does not exist.")

    # Apply CLI overrides (None means not provided)
    for section in cfg:
        for key in cfg[section]:
            v = getattr(args, key, None)
            if v is not None:
                cfg[section][key] = str(v)
    return cfg


def to_namespace(cfg: configparser.ConfigParser) -> SimpleNamespace:
    ns = SimpleNamespace(**combine_config(cfg))
    for c in vars(ns):
        setattr(ns, c, convert_string(getattr(ns, c)))
    return ns


def ensure_model_paths(cfg_ns: SimpleNamespace):
    if not getattr(cfg_ns, 'test', False):
        model_folder = os.path.join(cfg_ns.store_folder, cfg_ns.model_name)
        os.makedirs(model_folder, exist_ok=True)
        cfg_ns.model_folder = model_folder
        cfg_ns.config_path = os.path.join(model_folder, 'config.ini')
        cfg_ns.model_path = os.path.join(model_folder, 'model.pt')
        cfg_ns.tb_path = model_folder
    return cfg_ns


def save_config_file(cfg: configparser.ConfigParser, path: str, force: bool = False):
    mode = 'w' if force else 'x'
    with open(path, mode) as f:
        cfg.write(f)
