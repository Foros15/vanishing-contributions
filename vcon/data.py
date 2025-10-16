import torch
import torchvision
from torchvision.transforms import v2


def build_transforms(cfg):
    if cfg.dataset == 'cifar10':
        transform_mean = (0.49139968, 0.48215827, 0.44653124)
        transform_std = (0.24703233, 0.24348505, 0.26158768)
    elif cfg.dataset == 'cifar100':
        transform_mean = (0.5074, 0.4867, 0.4411)
        transform_std = (0.2011, 0.1987, 0.2025)
    elif cfg.dataset == 'imagenet1k':
        transform_mean = (0.485, 0.456, 0.406)
        transform_std = (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    transform_train_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
    ]
    if cfg.use_autoaugment:
        transform_train_list.append(v2.AutoAugment())
    if cfg.use_randaugment:
        transform_train_list.append(v2.RandAugment())
    transform_train_list.append(v2.Normalize(transform_mean, transform_std))

    transform_train = v2.Compose(transform_train_list)
    transform_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.Normalize(transform_mean, transform_std),
    ])
    return transform_train, transform_test


def build_datasets(cfg):
    transform_train, transform_test = build_transforms(cfg)
    if cfg.dataset == 'cifar10':
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root=cfg.dataset_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=cfg.dataset_root, train=False, download=True, transform=transform_test)
    elif cfg.dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root=cfg.dataset_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=cfg.dataset_root, train=False, download=True, transform=transform_test)
    elif cfg.dataset == 'imagenet1k':
        num_classes = 1000
        trainset = torchvision.datasets.ImageNet(root=cfg.dataset_root, split='train', transform=transform_train)
        testset = torchvision.datasets.ImageNet(root=cfg.dataset_root, split='val', transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    valset, testset = torch.utils.data.random_split(testset, [0.5, 0.5], generator=torch.Generator().manual_seed(42))
    return trainset, (valset, testset), num_classes


def build_dataloaders(cfg, trainset, valset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return trainloader, valloader, testloader


def build_mixup_cutmix(cfg, num_classes):
    mixup = v2.MixUp(alpha=cfg.mixup_alpha, num_classes=num_classes)
    if cfg.use_mixup_only:
        return mixup
    cutmix = v2.CutMix(num_classes=num_classes)
    return v2.RandomChoice([cutmix, mixup])
