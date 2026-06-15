from __future__ import annotations

import torch
import torch.nn.functional as F



@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def confidence(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    confidences = []
    for images, _ in data_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        max_probs = probs.max(dim=1).values
        confidences.append(max_probs)
    return torch.cat(confidences).mean().item()


@torch.no_grad()
def average_ensemble_accuracy(models_with_idx, data_loader, num_classes):
    device = next(models_with_idx[0][0].parameters()).device
    total = 0
    correct = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        avg_log_probs = torch.zeros((labels.size(0), num_classes), device=device)
        for model, _ in models_with_idx:
            model.eval()
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            avg_log_probs += torch.log(probs + 1e-12)
        avg_log_probs /= len(models_with_idx)
        avg_probs = torch.exp(avg_log_probs)
        predicted = avg_probs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def average_ensemble_confidence(models_with_idx, data_loader, num_classes):
    device = next(models_with_idx[0][0].parameters()).device
    total_conf_sum = 0.0
    total_samples = 0
    for images, _ in data_loader:
        images = images.to(device)
        avg_log_probs = torch.zeros((images.size(0), num_classes), device=device)
        for model, _ in models_with_idx:
            model.eval()
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            avg_log_probs += torch.log(probs + 1e-12)
        avg_log_probs /= len(models_with_idx)
        avg_probs = torch.exp(avg_log_probs)
        max_probs = avg_probs.max(dim=1).values
        total_conf_sum += max_probs.sum().item()
        total_samples += images.size(0)
    return (total_conf_sum / total_samples) if total_samples > 0 else 0.0


@torch.no_grad()
def average_learner_accuracy(models_with_idx, data_loader):
    accuracy_sum = 0.0
    for model, _ in models_with_idx:
        accuracy_sum += accuracy(model, data_loader)
    return accuracy_sum / len(models_with_idx)


@torch.no_grad()
def collect_preds_and_probs(models_with_idx, loader, num_classes, device=None):
    models = [m for m, _ in models_with_idx]
    ids = [idx for _, idx in models_with_idx]
    for m in models:
        m.eval()
    if device is None:
        device = next(models[0].parameters()).device
    all_preds = []
    all_probs = []
    for images, _labels in loader:
        images = images.to(device)
        logits = torch.stack([m(images) for m in models], dim=0)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
    return ids, torch.cat(all_preds, dim=1), torch.cat(all_probs, dim=1)


@torch.no_grad()
def pairwise_disagreement_matrix(models_with_idx, loader, num_classes, device=None):
    ids, preds, _ = collect_preds_and_probs(models_with_idx, loader, num_classes, device=device)
    m, _n = preds.shape
    d = torch.zeros((m, m), dtype=torch.float32)
    for i in range(m):
        for j in range(i + 1, m):
            dij = (preds[i] != preds[j]).float().mean()
            d[i, j] = d[j, i] = dij
    mean_pairwise = d.sum() / (m * (m - 1))
    return ids, d, float(mean_pairwise.item())


@torch.no_grad()
def variance_model_accuracy(models_with_idx, dataloader):
    accs = torch.tensor([accuracy(m, dataloader) for m, _ in models_with_idx], dtype=torch.float32)
    return float(accs.var(unbiased=False).item())



def get_models_no_oracle(updated_models, oracle_id=0):
    return [(m, mid) for (m, mid) in updated_models if mid != oracle_id]


@torch.no_grad()
def ensemble_val_acc_no_oracle(updated_models, val_loader, num_classes, oracle_id=0):
    models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
    return average_ensemble_accuracy(models_no_oracle, val_loader, num_classes)
