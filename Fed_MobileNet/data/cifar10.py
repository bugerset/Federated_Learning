from torchvision import datasets, transforms

def get_cifar10(root="./data", normalize = True, augment = False):

    # Normalize CIFAR-10 with below values
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = []

    if augment:
        # RandomCrop & HorizontalFlip
        train_tf += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

    train_tf.append(transforms.ToTensor())
    test_tf = [transforms.ToTensor()]

    if normalize:
        train_tf.append(transforms.Normalize(mean, std))
        test_tf.append(transforms.Normalize(mean, std))

    train_data = datasets.CIFAR10(
        root=root,   
        train=True, 
        download=True,
        transform=transforms.Compose(train_tf)
    )

    test_data  = datasets.CIFAR10(
        root=root, 
        train=False, 
        download=True,
        transform=transforms.Compose(test_tf)
    )
    
    return train_data, test_data