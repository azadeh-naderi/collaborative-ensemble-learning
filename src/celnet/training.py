from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .metrics import accuracy


def train_student_oracle(student, train_loader, optimizer , lr_scheduler=None):
    """Supervised step: CrossEntropy with true labels (Oracle)."""
    criterion = nn.CrossEntropyLoss()
    student.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        student_logits = student(images)
        loss = criterion(student_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return student


def train_student_kd(student, teacher, train_loader, optimizer , lr_scheduler=None):
    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(reduction="batchmean")

    student.train()
    teacher.eval()

    for images, _labels in train_loader:
        images = images.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)
            pseudo_labels = teacher_logits.argmax(dim=1)

        student_logits = student(images)

        hard_loss = criterion_hard(student_logits, pseudo_labels)
        soft_loss = criterion_soft(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
        )

        loss = (1 - alpha) * hard_loss + alpha * (temperature ** 2) * soft_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return student


def get_opt_sched(model_id: int, model: nn.Module):
    if model_id not in optimizers:

        if optimizer_type.lower() == "sgd":
            opt = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        elif optimizer_type.lower() == "adam":
            opt = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

        sch = None
        if use_scheduler:
            # milestones relative to total number of "round steps"
            m1 = max(1, int(0.55 * n_rounds))
            m2 = max(m1 + 1, int(0.80 * n_rounds))
            sch = torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=[m1, m2],
                gamma=gamma
            )

        optimizers[model_id] = opt
        schedulers[model_id] = sch

    return optimizers[model_id], schedulers[model_id]


@torch.no_grad()
def collect_preds_and_probs(models_with_idx, loader, num_classes, device=None):
    models = [m for m, _ in models_with_idx]
    ids = [idx for _, idx in models_with_idx]
    for m in models: m.eval()

    if device is None:
        device = next(models[0].parameters()).device

    all_preds = []
    all_probs = []

    for images, _labels in loader:
        images = images.to(device)

        logits = torch.stack([m(images) for m in models], dim=0)          # [M,B,C]
        probs  = torch.softmax(logits, dim=-1)                            # [M,B,C]
        preds  = probs.argmax(dim=-1)                                     # [M,B]

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())

    all_probs = torch.cat(all_probs, dim=1)   # [M,N,C]
    all_preds = torch.cat(all_preds, dim=1)   # [M,N]
    return ids, all_preds, all_probs


def distill_ensemble_to_student(ensemble_models, student_model, train_loader, val_loader, device, temperature=temperature, alpha=alpha, epochs=1, lr=0.01):

    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=momentum)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")


    student_model.train()

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # ---------------------------
            # Ensemble teacher prediction
            # ---------------------------
            with torch.no_grad():

                logits_sum = None

                for model, _ in ensemble_models:
                    model.eval()
                    logits = model(images)

                    if logits_sum is None:
                        logits_sum = logits
                    else:
                        logits_sum += logits

                teacher_logits = logits_sum / len(ensemble_models)

            # ---------------------------
            # Student forward
            # ---------------------------
            student_logits = student_model(images)

            # Hard loss
            loss_ce = ce_loss(student_logits, labels)

            # Soft loss
            loss_kd = kl_loss(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature ** 2)

            loss = alpha * loss_kd + (1 - alpha) * loss_ce

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = accuracy(student_model, val_loader)

        print(
            f"[Distill Epoch {epoch+1}/{epochs}] "
            f"Loss={total_loss/len(train_loader):.4f} "
            f"ValAcc={val_acc:.2f}"
        )

    return student_model
