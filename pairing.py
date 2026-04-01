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
def random_graph_pairing(tuple_models, degree=5, seed=None):
    models = [model for model, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]
    num_models = len(tuple_models)
    if num_models < 2:
        return []
    rng = random.Random(seed) if seed is not None else random
    g = nx.Graph()
    g.add_nodes_from(range(num_models))
    for i in range(num_models):
        possible_neighbors = [j for j in range(num_models) if j != i]
        chosen_neighbors = rng.sample(possible_neighbors, k=min(degree, len(possible_neighbors)))
        for j in chosen_neighbors:
            g.add_edge(i, j, weight=1.0)
    matching = nx.max_weight_matching(g, maxcardinality=True)
    return [((models[i], model_indices[i]), (models[j], model_indices[j])) for i, j in matching]


@torch.no_grad()
def random_graph_mwm_acc_pairing(tuple_models, val_loader, degree=5, seed=None):
    num_models = len(tuple_models)
    if num_models < 2:
        return []
    rng = random.Random(seed) if seed is not None else random
    models = [m for m, _ in tuple_models]
    ids = [idx for _, idx in tuple_models]
    accs = [float(accuracy(model, val_loader)) for model, _ in tuple_models]
    g = nx.Graph()
    g.add_nodes_from(range(num_models))
    for i in range(num_models):
        possible_neighbors = [j for j in range(num_models) if j != i]
        chosen_neighbors = rng.sample(possible_neighbors, k=min(degree, len(possible_neighbors)))
        for j in chosen_neighbors:
            if not g.has_edge(i, j):
                g.add_edge(i, j, weight=abs(accs[i] - accs[j]))
    matching = nx.max_weight_matching(g, maxcardinality=True)
    return [((models[i], ids[i]), (models[j], ids[j])) for i, j in matching]


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


def build_pairing_methods(updated_models, val_loader, val_splits, num_classes, degree, run_seed, batch_size, round_idx):
    return {
        "split": lambda: farthest_val_random_split(updated_models, val_splits, batch_size, run_seed),
        "euclidean": lambda: euclidean_distance_class_accuracy(updated_models, val_loader, num_classes),
        "max": lambda: max_difference_pairing(updated_models, val_loader),
        "mwm_classAcc": lambda: maximum_weight_matching_class_accuracy(updated_models, val_loader, num_classes),
        "mwm_acc": lambda: maximum_weight_matching_accuracy_difference(updated_models, val_loader),
        "random_graph": lambda: random_graph_pairing(updated_models, degree=degree, seed=run_seed),
        "random_graph_mwm_acc": lambda: random_graph_mwm_acc_pairing(updated_models, val_loader, degree=degree, seed=run_seed + round_idx),
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
