from vcon.config import default_config, build_arg_parser, load_and_merge_config, to_namespace, ensure_model_paths, save_config_file
from vcon.data import build_datasets, build_dataloaders, build_mixup_cutmix
from vcon.modeling import build_model
from vcon.trainer import Trainer


def main(argv=None):
    cfg_ini = default_config()
    parser = build_arg_parser(cfg_ini)
    args = parser.parse_args(args=argv)

    cfg_ini = load_and_merge_config(cfg_ini, args)
    cfg = to_namespace(cfg_ini)

    # Attach flags to namespace for convenience
    cfg.test = bool(args.test)
    cfg.force = bool(args.force)

    ensure_model_paths(cfg)

    # Persist config to model folder if training
    if not cfg.test and getattr(cfg, 'config_path', None):
        try:
            save_config_file(cfg_ini, cfg.config_path, force=cfg.force)
        except FileExistsError:
            # Not forcing; ignore
            pass

    # Data
    trainset, (valset, testset), num_classes = build_datasets(cfg)
    trainloader, valloader, testloader = build_dataloaders(cfg, trainset, valset, testset)
    print('training set length:', len(trainloader))
    print('validation set length:', len(valloader))
    print('test set length:', len(testloader))

    augment_op = build_mixup_cutmix(cfg, num_classes) if cfg.use_cutmix_mixup else None

    # Model
    model = build_model(cfg, num_classes)

    # Train/Test
    trainer = Trainer(cfg, model, (trainloader, valloader, testloader), num_classes, augment_op)
    return trainer.fit()


if __name__ == '__main__':
    main()
