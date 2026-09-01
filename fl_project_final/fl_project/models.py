import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_diagnostics():
    print("\n=== Device Diagnostics ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU VRAM: {total_mem:.2f} GB")
    print(f"Selected device: {device}")
    print("==========================\n")

print_device_diagnostics()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.f1 = nn.Linear(64 * 8 * 8, 128)
        self.f2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.f1(x))
        return self.f2(x)


def _client_labels(data):
    if hasattr(data, "indices") and hasattr(data, "dataset"):
        return [data.dataset.targets[i] for i in data.indices]
    return [int(data[i][1]) for i in range(len(data))]


def _class_weights(labels, num_classes=2):
    counts = [max(labels.count(c), 1) for c in range(num_classes)]
    total = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_local(model, data, epochs=2, lr=0.01, batch_size=32):
    model = model.to(device)
    model.train()
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    labels = _client_labels(data)
    weight = _class_weights(labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    return state


@torch.no_grad()
def evaluate_model(model, test_data, batch_size=1024):
    model.eval()
    loader = DataLoader(test_data, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()

    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item()
        probs = torch.softmax(out, dim=1)[:, 1]
        all_preds.append(out.argmax(1))
        all_labels.append(y)
        all_probs.append(probs)

    all_preds = torch.cat(all_preds).cpu().numpy().tolist()
    all_labels = torch.cat(all_labels).cpu().numpy().tolist()
    all_probs = torch.cat(all_probs).cpu().numpy().tolist()

    acc = round(100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels), 2)
    loss = round(total_loss / len(loader), 4)
    f1 = round(f1_score(all_labels, all_preds, zero_division=0), 4)
    auc = round(roc_auc_score(all_labels, all_probs), 4) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {"accuracy": acc, "loss": loss, "f1": f1, "auc": auc,
            "confusion_matrix": cm, "labels": all_labels,
            "probs": all_probs, "preds": all_preds}
