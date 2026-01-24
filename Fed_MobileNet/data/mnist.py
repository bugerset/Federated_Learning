from torchvision import datasets, transforms

def get_mnist(root="./data", normalize = True, augment = False):

    # Normalize MNIST with below values
    mean = (0.1307,)
    std  = (0.3081,)

    train_tf = []

    if augment:
        # RandomCrop & HorizontalFlip
        train_tf += [
            transforms.RandomCrop(32, padding=4),
        ]

    train_tf.append(transforms.ToTensor())
    test_tf = [transforms.ToTensor()]

    if normalize:
        train_tf.append(transforms.Normalize(mean, std))
        test_tf.append(transforms.Normalize(mean, std))

    train_data = datasets.MNIST(
        root=root,   
        train=True, 
        download=True,
        transform=transforms.Compose(train_tf)
    )

    test_data  = datasets.MNIST(
        root=root, 
        train=False, 
        download=True,
        transform=transforms.Compose(test_tf)
    )
    
    return train_data, test_data
