import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

NORMAL_CLASSES = [0]
DATA_ROOT = "./data"


def _binarize(targets, normal_classes):
    normal_set = set(normal_classes)
    return [0 if t in normal_set else 1 for t in targets]


class BinaryAnomalyCIFAR10(Dataset):
    def __init__(self, root=DATA_ROOT, train=True, normal_classes=None,
                 transform=None, download=True):
        self.normal_classes = normal_classes or NORMAL_CLASSES
        self.base = datasets.CIFAR10(root=root, train=train,
                                      download=download, transform=None)
        self.transform = transform
        self.targets = _binarize(self.base.targets, self.normal_classes)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_datasets(normal_classes=None, root=DATA_ROOT, download=True):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                              [0.2470, 0.2435, 0.2616]),
    ])
    train_data = BinaryAnomalyCIFAR10(root=root, train=True,
                                       normal_classes=normal_classes,
                                       transform=tf, download=download)
    test_data = BinaryAnomalyCIFAR10(root=root, train=False,
                                      normal_classes=normal_classes,
                                      transform=tf, download=download)
    return train_data, test_data


def make_clients(train_data, num_clients=5, alpha=1.0, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(train_data.targets)
    num_classes = len(set(targets.tolist()))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        split = np.split(idx_c, cuts)
        for client_id, idx_chunk in enumerate(split):
            client_indices[client_id].extend(idx_chunk.tolist())

    return [Subset(train_data, idxs) for idxs in client_indices]


def client_class_distribution(clients, train_data):
    targets = np.array(train_data.targets)
    num_classes = len(set(targets.tolist()))
    counts = np.zeros((len(clients), num_classes), dtype=int)
    for i, c in enumerate(clients):
        for idx in c.indices:
            counts[i, targets[idx]] += 1
    return counts
