from __future__ import annotations

import random
from typing import List, Tuple

import networkx as nx
import torch

from .metrics import accuracy


@torch.no_grad()
def per_class_accuracy_vector(model, val_loader, num_classes):
    device = next(model.parameters()).device
    correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    total = torch.zeros(num_classes, dtype=torch.long, device=device)
    model.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        total += torch.bincount(labels, minlength=num_classes)
        correct += torch.bincount(labels[preds == labels], minlength=num_classes)
    acc = torch.zeros(num_classes, dtype=torch.float32, device=device)
    nonzero = total > 0
    acc[nonzero] = correct[nonzero].float() / total[nonzero].float()
    return acc


@torch.no_grad()
def get_model_feature_vector(model, model_idx, val_loader, num_classes):
    if model_idx == 0:
        device = next(model.parameters()).device
        return torch.ones(num_classes, dtype=torch.float32, device=device)
    return per_class_accuracy_vector(model, val_loader, num_classes)


@torch.no_grad()
def build_feature_vectors(tuple_models, val_loader, num_classes):
    return torch.stack([
        get_model_feature_vector(model, idx, val_loader, num_classes)
        for model, idx in tuple_models
    ])


@torch.no_grad()
def compute_pairwise_euclidean_distances(feature_vectors):
    num_models = feature_vectors.shape[0]
    device = feature_vectors.device
    distances = torch.zeros((num_models, num_models), dtype=torch.float32, device=device)
    distance_list = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            dist = torch.norm(feature_vectors[i] - feature_vectors[j], p=2)
            distances[i, j] = distances[j, i] = dist
            distance_list.append((dist.item(), i, j))
    return distances, distance_list


def greedy_farthest_pairing(distance_list, tuple_models):
    models = [m for m, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]
    distance_list = sorted(distance_list, reverse=True)
    paired = set()
    pairs = []
    for dist, i, j in distance_list:
        if i not in paired and j not in paired:
            paired.add(i)
            paired.add(j)
            pairs.append(((models[i], model_indices[i]), (models[j], model_indices[j])))
        if len(paired) == len(models):
            break
    return pairs


@torch.no_grad()
def euclidean_distance_class_accuracy(tuple_models, val_loader, num_classes):
    feature_vectors = build_feature_vectors(tuple_models, val_loader, num_classes)
    _, distance_list = compute_pairwise_euclidean_distances(feature_vectors)
    return greedy_farthest_pairing(distance_list, tuple_models)


@torch.no_grad()
def maximum_weight_matching_class_accuracy(tuple_models, val_loader, num_classes):
    models = [m for m, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]
    feature_vectors = build_feature_vectors(tuple_models, val_loader, num_classes)
    distances, _ = compute_pairwise_euclidean_distances(feature_vectors)
    g = nx.Graph()
    num_models = len(tuple_models)
    for i in range(num_models):
        g.add_node(i)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            g.add_edge(i, j, weight=float(distances[i, j].item()))
    matching = nx.max_weight_matching(g, maxcardinality=True)
    return [((models[i], model_indices[i]), (models[j], model_indices[j])) for i, j in matching]


@torch.no_grad()
def maximum_weight_matching_accuracy_difference(tuple_models, val_loader):
    models = [model for model, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]
    accuracies = []
    for model, idx in tuple_models:
        acc = 1.0 if idx == 0 else float(accuracy(model, val_loader))
        accuracies.append(acc)
    g = nx.Graph()
    num_models = len(tuple_models)
    for i in range(num_models):
        g.add_node(i)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            g.add_edge(i, j, weight=abs(accuracies[i] - accuracies[j]))
    matching = nx.max_weight_matching(g, maxcardinality=True)
    return [((models[i], model_indices[i]), (models[j], model_indices[j])) for i, j in matching]



@torch.no_grad()
def compute_feature_vector_accuracy(model, val_splits, batch_size, run_seed):
    feature_vector = torch.zeros(len(val_splits))
    for j, subset in enumerate(val_splits):
        generator = torch.Generator()
        generator.manual_seed(run_seed)
        loader = torch.utils.data.DataLoader(subset, generator=generator, batch_size=batch_size, shuffle=False)
        feature_vector[j] = accuracy(model, loader)
    return feature_vector


@torch.no_grad()
def farthest_val_random_split(tuple_models, val_splits, batch_size, run_seed):
    models = [model for model, idx in tuple_models]
    initial_indices = [idx for model, idx in tuple_models]
    feature_vectors = [compute_feature_vector_accuracy(model, val_splits, batch_size, run_seed) for model in models]
    feature_vectors = torch.stack(feature_vectors)
    num_models = len(models)
    distance_list = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            dist = torch.norm(feature_vectors[i] - feature_vectors[j], p=2)
            distance_list.append((dist.item(), i, j))
    distance_list.sort(reverse=True)
    paired = set()
    pairs = []
    for dist, i, j in distance_list:
        if i not in paired and j not in paired:
            paired.update([i, j])
            pairs.append(((models[i], initial_indices[i]), (models[j], initial_indices[j])))
        if len(paired) == num_models:
            break
    return pairs


def make_fixed_friend_groups_by_id(tuple_models, group_size=5, seed=None):
    """
    Create fixed random friend groups using model IDs only.
    Call this once before the round loop.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2.")

    ids = [idx for _, idx in tuple_models]

    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(ids)

    group_ids = []
    for i in range(0, len(ids), group_size):
        group_ids.append(ids[i:i + group_size])

    return group_ids


def materialize_groups_from_ids(updated_models, group_ids):
    """
    Convert fixed group IDs into current (model, idx) tuples.
    """
    id_to_model = {idx: (model, idx) for model, idx in updated_models}

    groups = []
    for gid_group in group_ids:
        group = [id_to_model[idx] for idx in gid_group if idx in id_to_model]
        if len(group) > 0:
            groups.append(group)

    return groups


def pair_fixed_groups_random(groups, seed=None):
    """
    Random pairing, but only inside each fixed friend group.
    """
    rng = random.Random(seed) if seed is not None else random
    pairs = []

    for group in groups:
        if len(group) < 2:
            continue

        members = list(group)
        rng.shuffle(members)

        for i in range(0, len(members) - 1, 2):
            pairs.append((members[i], members[i + 1]))

    return pairs


@torch.no_grad()
def pair_fixed_groups_mwm_acc_diff(groups, val_loader):
    """
    Pair only inside each fixed friend group, using maximum-weight matching
    with edge weight = absolute accuracy difference.
    """
    pairs = []

    for group in groups:
        if len(group) < 2:
            continue

        models = [m for m, _ in group]
        ids = [idx for _, idx in group]

        accs = []
        for model, idx in group:
            if idx == 0:
                acc = 100.0
            else:
                acc = float(accuracy(model, val_loader))
            accs.append(acc)

        G = nx.Graph()
        G.add_nodes_from(range(len(group)))

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                weight = abs(accs[i] - accs[j])
                G.add_edge(i, j, weight=weight)

        matching = nx.max_weight_matching(G, maxcardinality=True)

        for i, j in matching:
            pairs.append(((models[i], ids[i]), (models[j], ids[j])))

    return pairs


@torch.no_grad()
def pair_fixed_groups_acc_diff(groups, val_loader):

    pairs = []

    for group in groups:
        if len(group) < 2:
            continue

        # Compute accuracies for members of this group
        members_with_acc = []
        for model, idx in group:
            if idx == 0:
                acc = 100.0   # oracle handling, same as your current code
            else:
                acc = float(accuracy(model, val_loader))
            members_with_acc.append((model, idx, acc))

        # Sort by accuracy from low to high
        members_with_acc.sort(key=lambda x: x[2])

        # Pair lowest with highest, next-lowest with next-highest, ...
        n = len(members_with_acc)
        for i in range(n // 2):
            low_model, low_id, low_acc = members_with_acc[i]
            high_model, high_id, high_acc = members_with_acc[n - 1 - i]

            pairs.append(((low_model, low_id), (high_model, high_id)))

    return pairs


@torch.no_grad()
def random_regular_graph_maxdiff_pairing(tuple_models, val_loader, degree, seed=None):
    """
    Include oracle in a true random d-regular graph, then choose disjoint pairs
    greedily by largest accuracy difference among allowed graph edges.
    """
    num_models = len(tuple_models)
    if num_models < 2:
        return []

    if degree >= num_models:
        raise ValueError(
            f"degree must be < num_models, got degree={degree}, num_models={num_models}"
        )

    if (num_models * degree) % 2 != 0:
        raise ValueError(
            f"A {degree}-regular graph on {num_models} nodes cannot exist because "
            f"num_models * degree must be even."
        )

    models = [m for m, _ in tuple_models]
    ids = [idx for _, idx in tuple_models]

    accs = []
    for model, idx in tuple_models:
        if idx == 0:
            accs.append(100.0)
        else:
            accs.append(float(accuracy(model, val_loader)))

    G = nx.random_regular_graph(d=degree, n=num_models, seed=seed)

    edge_list = []
    for i, j in G.edges():
        diff = abs(accs[i] - accs[j])
        edge_list.append((diff, i, j))

    edge_list.sort(reverse=True)

    paired = set()
    pairs = []

    for diff, i, j in edge_list:
        if i not in paired and j not in paired:
            paired.add(i)
            paired.add(j)
            pairs.append(((models[i], ids[i]), (models[j], ids[j])))

    return pairs

@torch.no_grad()
def random_regular_graph_uniform_pairing(tuple_models, degree, seed=None):
    """
    Random 3-regular graph + uniform random neighbor matching.

    Steps:
    1. Build random d-regular graph
    2. Randomly iterate nodes
    3. Each node picks a random available neighbor
    4. Form disjoint pairs

    Returns:
        [((model_a, idx_a), (model_b, idx_b)), ...]
    """
    num_models = len(tuple_models)
    if num_models < 2:
        return []

    if degree >= num_models:
        raise ValueError(f"degree must be < num_models")

    if (num_models * degree) % 2 != 0:
        raise ValueError("num_models * degree must be even for regular graph")

    rng = random.Random(seed)

    models = [m for m, _ in tuple_models]
    ids = [idx for _, idx in tuple_models]

    # Step 1: build random 3-regular graph
    G = nx.random_regular_graph(d=degree, n=num_models, seed=seed)

    # Step 2: shuffle node order
    nodes = list(range(num_models))
    rng.shuffle(nodes)

    paired = set()
    pairs = []

    # Step 3: greedy random neighbor matching
    for i in nodes:
        if i in paired:
            continue

        # available neighbors that are not yet paired
        neighbors = [j for j in G.neighbors(i) if j not in paired]

        if len(neighbors) == 0:
            continue  # no available neighbor left

        # pick one neighbor uniformly at random
        j = rng.choice(neighbors)

        paired.add(i)
        paired.add(j)

        pairs.append(((models[i], ids[i]), (models[j], ids[j])))

    return pairs

def build_pairing_methods(updated_models, val_loader, val_splits, num_classes, degree, run_seed, batch_size, round_idx, fixed_group_ids=None):
    return {
        "split": lambda: farthest_val_random_split(updated_models, val_splits, batch_size, run_seed),
        "euclidean": lambda: euclidean_distance_class_accuracy(updated_models, val_loader, num_classes),
        "max": lambda: max_difference_pairing(updated_models, val_loader),
        "mwm_classAcc": lambda: maximum_weight_matching_class_accuracy(updated_models, val_loader, num_classes),
        "mwm_acc": lambda: maximum_weight_matching_accuracy_difference(updated_models, val_loader),

        "friend_random": lambda: pair_fixed_groups_random(groups=materialize_groups_from_ids(updated_models, fixed_group_ids), seed=run_seed + round_idx),
        "friend_mwm_acc_diff": lambda: pair_fixed_groups_mwm_acc_diff(groups=materialize_groups_from_ids(updated_models, fixed_group_ids), val_loader=val_loader),
        "friend_acc_diff": lambda: pair_fixed_groups_acc_diff(groups=materialize_groups_from_ids(updated_models, fixed_group_ids), val_loader=val_loader),

        "random_3reg_max": lambda: random_regular_graph_maxdiff_pairing(tuple_models=updated_models, val_loader=val_loader, degree=degree, seed=run_seed + round_idx),
        "random_3reg_uniform": lambda: random_regular_graph_uniform_pairing(tuple_models=updated_models, degree=degree, seed=run_seed + round_idx),
    }


@torch.no_grad()
def max_difference_pairing(tuple_models, val_loader):
    models = [m for m, _ in tuple_models]
    initial_indices = [idx for _, idx in tuple_models]
    acc_list = []
    for m, idx in tuple_models:
        acc_list.append(100.0 if idx == 0 else float(accuracy(m, val_loader)))
    accuracies = torch.tensor(acc_list)
    sorted_indices = torch.argsort(accuracies)
    pairs = []
    n = len(models)
    for i in range(n // 2):
        model_1 = models[sorted_indices[i].item()]
        model_1_idx = initial_indices[sorted_indices[i].item()]
        model_2 = models[sorted_indices[n - 1 - i].item()]
        model_2_idx = initial_indices[sorted_indices[n - 1 - i].item()]
        pairs.append(((model_1, model_1_idx), (model_2, model_2_idx)))
    return pairs
