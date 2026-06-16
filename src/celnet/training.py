from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .metrics import accuracy


def train_student_oracle(student, train_loader, optimizer, device, lr_scheduler=None):
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


def train_student_kd(student, teacher, train_loader, optimizer, device, temperature, alpha, lr_scheduler=None):
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


def train_pair_pom(m1, m2, train_loader, opt1, opt2, device, temperature, alpha, lr_scheduler1=None, lr_scheduler2=None):
    """
    POM symmetric training: both models exchange soft predictions and train on
    each other's outputs. Forward passes for both happen before either backward
    pass so neither model trains on already-updated weights.
    """
    criterion_soft = nn.KLDivLoss(reduction="batchmean")
    criterion_hard = nn.CrossEntropyLoss()
    m1.train()
    m2.train()
    for images, _labels in train_loader:
        images = images.to(device)
        with torch.no_grad():
            logits1 = m1(images)
            logits2 = m2(images)
        out1 = m1(images)
        hard_loss1 = criterion_hard(out1, logits2.argmax(dim=1))
        soft_loss1 = criterion_soft(
            F.log_softmax(out1 / temperature, dim=1),
            F.softmax(logits2 / temperature, dim=1),
        )
        loss1 = (1 - alpha) * hard_loss1 + alpha * (temperature ** 2) * soft_loss1
        opt1.zero_grad()
        loss1.backward()
        opt1.step()
        out2 = m2(images)
        hard_loss2 = criterion_hard(out2, logits1.argmax(dim=1))
        soft_loss2 = criterion_soft(
            F.log_softmax(out2 / temperature, dim=1),
            F.softmax(logits1 / temperature, dim=1),
        )
        loss2 = (1 - alpha) * hard_loss2 + alpha * (temperature ** 2) * soft_loss2
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
    if lr_scheduler1 is not None:
        lr_scheduler1.step()
    if lr_scheduler2 is not None:
        lr_scheduler2.step()
    return m1, m2


class OptimizerRegistry:
    def __init__(self):
        self._optimizers = {}
        self._schedulers = {}

    def get(self, model_id, model, optimizer_type, lr, momentum, weight_decay, use_scheduler, n_rounds, gamma):
        if model_id not in self._optimizers:
            if optimizer_type.lower() == "sgd":
                opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type.lower() == "adam":
                opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
            sch = None
            if use_scheduler:
                m1 = max(1, int(0.55 * n_rounds))
                m2 = max(m1 + 1, int(0.80 * n_rounds))
                sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m1, m2], gamma=gamma)
            self._optimizers[model_id] = opt
            self._schedulers[model_id] = sch
        return self._optimizers[model_id], self._schedulers[model_id]


def distill_ensemble_to_student(ensemble_models, student_model, train_loader, val_loader, device, temperature=4.0, alpha=0.9, epochs=100, lr=0.01):
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    student_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                logits_sum = sum(model(images) for model, _ in ensemble_models)
                teacher_logits = logits_sum / len(ensemble_models)
            student_logits = student_model(images)
            loss_ce = ce_loss(student_logits, labels)
            loss_kd = kl_loss(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
            ) * (temperature ** 2)
            loss = alpha * loss_kd + (1 - alpha) * loss_ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = accuracy(student_model, val_loader)
        print(f"[Distill Epoch {epoch+1}/{epochs}] Loss={total_loss/len(train_loader):.4f} ValAcc={val_acc:.2f}")
    return student_model
