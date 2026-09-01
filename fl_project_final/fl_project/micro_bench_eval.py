import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from models import Net, device
from dataset import get_datasets

train_data, test_data = get_datasets(root="./data", download=False)
model = Net().to(device)

def original_evaluate():
    t0 = time.time()
    model.eval()
    loader = DataLoader(test_data, batch_size=128)
    loss_fn = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item()
        probs = torch.softmax(out, dim=1)[:, 1]
        all_preds.extend(out.argmax(1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    t1 = time.time()
    print(f"Original Eval (batch=128): {t1-t0:.2f}s")

def optimized_evaluate():
    t0 = time.time()
    model.eval()
    loader = DataLoader(test_data, batch_size=1024)
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
    t1 = time.time()
    print(f"Optimized Eval (batch=1024): {t1-t0:.2f}s")

with torch.no_grad():
    original_evaluate()
    optimized_evaluate()
