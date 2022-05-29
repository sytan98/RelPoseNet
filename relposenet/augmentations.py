import torchvision.transforms as transforms


def get_imagenet_mean_std():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std


def get_augmentations():
    train_aug = train_augmentations()
    val_aug = eval_augmentations()
    return train_aug, val_aug


def train_augmentations():
    # mean, std = get_imagenet_mean_std()
    mean, std = [0.7208, 0.6812, 0.6156], [0.2159, 0.2165, 0.2228]
    transform = transforms.Compose([transforms.Resize(size=224),
                                    transforms.CenterCrop(size=224),
                                    # transforms.ColorJitter(brightness=0.4,
                                    #                        contrast=0.4,
                                    #                        saturation=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    return transform


def eval_augmentations():
    # mean, std = get_imagenet_mean_std()
    mean, std = [0.7208, 0.6812, 0.6156], [0.2159, 0.2165, 0.2228]
    transform = transforms.Compose([transforms.Resize(size=224),
                                    transforms.CenterCrop(size=224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    return transform
