"""What does one update change? Used by switch_peers_celnet_pairing.py --mechanism at each oracle CE update.

For an update from state S0 to S1 (the real CE epoch, or the counterfactual KD epoch from the same S0), on validation
images and on fixed training images:
  * images are grouped by S0: correct (A), wrong but agreeing with the peer (B), wrong and not agreeing (C);
    broken = A images S1 gets wrong, fixed_* = B / C images S1 gets right (all in % of the split), so
    accuracy change = fixed_peer_agree + fixed_peer_disagree - broken,
  * changed = predictions that differ between S0 and S1, changed_to_peer = those whose new prediction is the peer's,
  * kl = mean KL(S0 || S1) of the softmax outputs, conf_change = change of the mean top-class probability,
  * weight change of the backbone and of the final layer, relative to their S0 norm,
  * head swap (validation only): accuracy of S1's backbone with S0's final layer, and of S0's backbone with S1's.
The peer is the counterfactual teacher (the most accurate other learner at the start of the round).
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from switch_common import kd_loss, predict
from switch_run_student import train_epoch

HEAD_PREFIX = "base_model.fc."


def kd_twin(student, opt, teacher, loader, device, alpha, temperature):
    """Copy of the student after one KD epoch from its current weights and optimizer state; the student is untouched."""
    twin = copy.deepcopy(student)
    twin_opt = torch.optim.SGD(twin.parameters(), lr=opt.param_groups[0]["lr"], momentum=0.9, weight_decay=1e-4)
    # deepcopy: load_state_dict would otherwise share the momentum buffers with the real optimizer
    twin_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    train_epoch(twin, teacher, loader, twin_opt, device, "kd", alpha, temperature)
    return twin


CE_VARIANTS = ("ce", "ce_head", "ce_lowlr")
KD_VARIANTS = ("kd", "kd_head", "kd_lowlr")


def update(model, opt, loader, device, variant, low_lr, teacher=None, alpha=0.9, temperature=4.0):
    """One epoch of the given variant, in place. The loss is CE on the true labels (ce*) or the KD loss from
    `teacher` (kd*, same loss as switch_run_student.train_epoch).
    *        : the normal update.
    *_head   : only the final layer learns; the backbone is frozen, including BatchNorm statistics (eval mode), so the
               features cannot change. SGD skips parameters without a gradient, so their momentum is left as it is.
    *_lowlr  : the normal update with the learning rate lowered to low_lr for this epoch only."""
    loss_type, _, scope = variant.partition("_")
    if loss_type not in ("ce", "kd") or scope not in ("", "head", "lowlr"):
        raise ValueError(variant)
    if loss_type == "kd" and teacher is None:
        raise ValueError(f"{variant} needs a teacher")
    if scope == "":
        train_epoch(model, teacher, loader, opt, device, loss_type, alpha, temperature)
    elif scope == "lowlr":
        saved = [g["lr"] for g in opt.param_groups]
        for g in opt.param_groups:
            g["lr"] = low_lr
        try:
            train_epoch(model, teacher, loader, opt, device, loss_type, alpha, temperature)
        finally:
            for g, lr in zip(opt.param_groups, saved):
                g["lr"] = lr
    else:
        trainable = {n: p.requires_grad for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(HEAD_PREFIX))
        model.eval()
        try:
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                if loss_type == "ce":
                    loss = F.cross_entropy(logits, y)
                else:
                    with torch.no_grad():
                        teacher_logits = teacher(x)
                    loss = alpha * kd_loss(logits, teacher_logits, temperature)
                    if alpha < 1.0:
                        loss = loss + (1.0 - alpha) * F.cross_entropy(logits, teacher_logits.argmax(dim=1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        finally:
            for n, p in model.named_parameters():
                p.requires_grad_(trainable[n])
            opt.zero_grad(set_to_none=True)


def update_twin(student, opt, loader, device, variant, low_lr, teacher=None, alpha=0.9, temperature=4.0):
    """Copy of the student after one epoch of the given variant; the student and its optimizer are untouched."""
    twin = copy.deepcopy(student)
    twin_opt = torch.optim.SGD(twin.parameters(), lr=opt.param_groups[0]["lr"], momentum=0.9, weight_decay=1e-4)
    twin_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    update(twin, twin_opt, loader, device, variant, low_lr, teacher, alpha, temperature)
    return twin


def ce_update(model, opt, loader, device, variant, low_lr):
    update(model, opt, loader, device, variant, low_lr)


def ce_twin(student, opt, loader, device, variant, low_lr):
    return update_twin(student, opt, loader, device, variant, low_lr)


def _mix(backbone_state, head_state):
    mixed = dict(backbone_state)
    mixed.update({k: v for k, v in head_state.items() if k.startswith(HEAD_PREFIX)})
    return mixed


def _relative_change(before, after, names):
    num = sum(float(((after[n].double() - before[n].double()) ** 2).sum()) for n in names)
    den = sum(float((before[n].double() ** 2).sum()) for n in names)
    return (num / den) ** 0.5 if den > 0 else 0.0


@torch.no_grad()
def describe_updates(before_state, arms, peer, loaders, scratch, device):
    """before_state: S0 state_dict; arms: {name: model after the update}; loaders: {split: loader}.
    scratch: a spare model of the same architecture, used for S0 and the head swaps."""
    out = {}
    scratch.load_state_dict(before_state)
    ref = {}
    for split, loader in loaders.items():
        logits0, y = predict(scratch, loader, device)
        p0 = logits0.argmax(dim=1)
        peer_pred = predict(peer, loader, device)[0].argmax(dim=1)
        a = p0.eq(y)
        b = ~a & p0.eq(peer_pred)
        c = ~a & ~p0.eq(peer_pred)
        out[f"{split}_acc_before"] = 100.0 * a.float().mean().item()
        out[f"{split}_share_wrong_agree_peer"] = 100.0 * b.float().mean().item()
        out[f"{split}_share_wrong_disagree_peer"] = 100.0 * c.float().mean().item()
        out[f"{split}_agree_peer_before"] = 100.0 * p0.eq(peer_pred).float().mean().item()
        ref[split] = (logits0, p0, y, peer_pred, a, b, c)

    params = [n for n, _ in scratch.named_parameters()]
    head = [n for n in params if n.startswith(HEAD_PREFIX)]
    backbone = [n for n in params if not n.startswith(HEAD_PREFIX)]
    for arm, model in arms.items():
        for split, loader in loaders.items():
            logits0, p0, y, peer_pred, a, b, c = ref[split]
            logits1 = predict(model, loader, device)[0]
            p1 = logits1.argmax(dim=1)
            ok = p1.eq(y)
            changed = ~p1.eq(p0)
            prob0, prob1 = logits0.softmax(dim=1), logits1.softmax(dim=1)
            pct = lambda mask: 100.0 * mask.float().mean().item()
            out.update({
                f"{arm}_{split}_acc_after": pct(ok),
                f"{arm}_{split}_broken": pct(a & ~ok),
                f"{arm}_{split}_fixed_peer_agree": pct(b & ok),
                f"{arm}_{split}_fixed_peer_disagree": pct(c & ok),
                f"{arm}_{split}_changed": pct(changed),
                f"{arm}_{split}_changed_to_peer": pct(changed & p1.eq(peer_pred)),
                f"{arm}_{split}_agree_peer_after": pct(p1.eq(peer_pred)),
                f"{arm}_{split}_kl": (prob0 * (logits0.log_softmax(dim=1) - logits1.log_softmax(dim=1)))
                .sum(dim=1).mean().item(),
                f"{arm}_{split}_conf_change": (prob1.max(dim=1).values - prob0.max(dim=1).values).mean().item(),
            })
        after = {k: v.detach() for k, v in model.state_dict().items()}
        out[f"{arm}_weight_change_backbone"] = _relative_change(before_state, after, backbone)
        out[f"{arm}_weight_change_head"] = _relative_change(before_state, after, head)
        val = loaders["val"]
        y_val = ref["val"][2]
        scratch.load_state_dict(_mix(after, before_state))
        out[f"{arm}_val_acc_new_backbone_old_head"] = 100.0 * predict(scratch, val, device)[0].argmax(1).eq(y_val) \
            .float().mean().item()
        scratch.load_state_dict(_mix(before_state, after))
        out[f"{arm}_val_acc_old_backbone_new_head"] = 100.0 * predict(scratch, val, device)[0].argmax(1).eq(y_val) \
            .float().mean().item()
    return out
