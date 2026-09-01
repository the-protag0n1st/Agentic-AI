import time
import copy
from models import Net, train_local, device
from dataset import get_datasets, make_clients
from torch.utils.data import DataLoader

train_data, _ = get_datasets(root="./data", download=False)
clients = make_clients(train_data, num_clients=5, alpha=1.0, seed=42)
gm = Net().to(device)

def test_original():
    t0 = time.time()
    for c in clients:
        train_local(copy.deepcopy(gm), c, epochs=1, lr=0.01, batch_size=32)
    t1 = time.time()
    print(f"Original (1 epoch): {t1-t0:.2f}s")

def test_reused_loader():
    loaders = [DataLoader(c, batch_size=32, shuffle=True) for c in clients]
    t0 = time.time()
    for l, c in zip(loaders, clients):
        model = copy.deepcopy(gm).to(device)
        model.train()
        import torch.optim as optim
        import torch.nn as nn
        from models import _client_labels, _class_weights
        labels = _client_labels(c)
        weight = _class_weights(labels).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        for _ in range(1):
            for x, y in l:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
    t1 = time.time()
    print(f"Reused loader (1 epoch): {t1-t0:.2f}s")

def test_batch_128():
    t0 = time.time()
    for c in clients:
        train_local(copy.deepcopy(gm), c, epochs=1, lr=0.01, batch_size=128)
    t1 = time.time()
    print(f"Batch 128 (1 epoch): {t1-t0:.2f}s")

test_original()
test_reused_loader()
test_batch_128()
