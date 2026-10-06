from torchvision import transforms  # type: ignore

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform():
    """Used for validation, test and later the API: no randomness."""
    return transforms.Compose([
        transforms.Resize((32, 32)),        # any uploaded image -> CIFAR size
        transforms.ToTensor(),              # pixels 0-255 -> numbers 0-1
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform():
    """Training adds small random changes so the model generalises (Step 9 expands this)."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])