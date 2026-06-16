from __future__ import annotations

import random
from typing import List, Tuple

import networkx as nx
import torch

from .metrics import accuracy

#-----------pairing---------------------

@torch.no_grad()
def max_difference_pairing(tuple_models, val_loader):

    models = [m for m, _ in tuple_models]
    initial_indices = [idx for _, idx in tuple_models]
    
    acc_list = []
    for m, idx in tuple_models:
        if idx == 0:
            acc_list.append(100.0)  # oracle
        else:
            acc_list.append(float(accuracy(m, val_loader)))

    accuracies = torch.tensor(acc_list)
    sorted_indices = torch.argsort(accuracies)

    pairs = []
    n = len(models)
    for i in range(n // 2):
        model_1 = models[sorted_indices[i].item()]           # Lowest accuracy model
        model_1_idx = initial_indices[sorted_indices[i].item()]
        model_2 = models[sorted_indices[n - 1 - i].item()]   # Highest accuracy model
        model_2_idx = initial_indices[sorted_indices[n - 1 - i].item()]
        pairs.append(((model_1, model_1_idx), (model_2, model_2_idx)))

    return pairs



@torch.no_grad()
def per_class_accuracy_vector(model, val_loader, num_classes):

    device = next(model.parameters()).device
    correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    total   = torch.zeros(num_classes, dtype=torch.long, device=device)

    model.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        # Count total per class in batch
        total += torch.bincount(labels, minlength=num_classes)

        # Count correct per class in batch
        correct += torch.bincount(labels[preds == labels], minlength=num_classes)

    # Avoid div-by-zero
    acc = torch.zeros(num_classes, dtype=torch.float32, device=device)
    nonzero = total > 0
    acc[nonzero] = correct[nonzero].float() / total[nonzero].float()

    return acc


@torch.no_grad()
def get_model_feature_vector(model, model_idx, val_loader, num_classes):
    """
    Return the feature vector for one model.
    If model_idx == 0, treat it as the oracle and return all-ones vector.
    """
    if model_idx == 0:
        device = next(model.parameters()).device
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    return per_class_accuracy_vector(model, val_loader, num_classes)


@torch.no_grad()
def build_feature_vectors(tuple_models, val_loader, num_classes):
    """
    Build stacked feature vectors for all models.
    Returns tensor of shape [num_models, num_classes].
    """
    feature_vectors = []

    for model, idx in tuple_models:
        vec = get_model_feature_vector(model, idx, val_loader, num_classes)
        feature_vectors.append(vec)

    return torch.stack(feature_vectors)


@torch.no_grad()
def compute_pairwise_euclidean_distances(feature_vectors):
    """
    Compute full symmetric pairwise Euclidean distance matrix.
    Returns:
        distances: [num_models, num_models]
        distance_list: list of (distance, i, j)
    """
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
    """
    Greedy farthest-first pairing based on a sorted distance list.
    """
    models = [m for m, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]

    distance_list = sorted(distance_list, reverse=True)

    paired = set()
    pairs = []

    for dist, i, j in distance_list:
        if i not in paired and j not in paired:
            paired.add(i)
            paired.add(j)
            pairs.append(((models[i], model_indices[i]),
                          (models[j], model_indices[j])))

        if len(paired) == len(models):
            break

    return pairs


@torch.no_grad()
def euclidean_distance_class_accuracy(tuple_models, val_loader, num_classes):
    """
    Pair models using greedy farthest-first matching based on Euclidean distance
    between per-class accuracy vectors.
    """
    feature_vectors = build_feature_vectors(tuple_models, val_loader, num_classes)
    distances, distance_list = compute_pairwise_euclidean_distances(feature_vectors)
    pairs = greedy_farthest_pairing(distance_list, tuple_models)
    return pairs


@torch.no_grad()
def maximum_weight_matching_class_accuracy(tuple_models, val_loader, num_classes):
    """
    Pair models using maximum-weight matching based on Euclidean distance
    between per-class accuracy vectors.

    - Each model can appear in at most one pair
    - Edge weight = Euclidean distance between model feature vectors
    - idx == 0 is treated as oracle by your existing feature functions
    """
    models = [m for m, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]

    # 1) Build feature vectors
    feature_vectors = build_feature_vectors(tuple_models, val_loader, num_classes)

    # 2) Compute pairwise distances
    distances, _ = compute_pairwise_euclidean_distances(feature_vectors)

    # 3) Build weighted graph
    G = nx.Graph()
    num_models = len(tuple_models)

    for i in range(num_models):
        G.add_node(i)

    for i in range(num_models):
        for j in range(i + 1, num_models):
            G.add_edge(i, j, weight=float(distances[i, j].item()))

    # 4) Maximum weight matching
    matching = nx.max_weight_matching(G, maxcardinality=True)

    # 5) Convert matching to your pair format
    pairs = []
    for i, j in matching:
        pairs.append(((models[i], model_indices[i]),
                      (models[j], model_indices[j])))

    return pairs

@torch.no_grad()
def maximum_weight_matching_accuracy_difference(tuple_models, val_loader):
    """
    Pair models using maximum-weight matching based on absolute accuracy difference.

    Edge weight between model i and j:
        |acc_i - acc_j|

    Returns:
        pairs in the same format as your other pairing strategies:
        [((model_a, idx_a), (model_b, idx_b)), ...]
    """
    models = [model for model, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]

    # Compute accuracies in tuple_models order
    accuracies = []
    for model, idx in tuple_models:
        if idx == 0:
            acc = 100.0   # oracle
        else:
            acc = float(accuracy(model, val_loader))
        accuracies.append(acc)

    # Build weighted graph
    G = nx.Graph()
    num_models = len(tuple_models)

    for i in range(num_models):
        G.add_node(i)

    for i in range(num_models):
        for j in range(i + 1, num_models):
            weight = abs(accuracies[i] - accuracies[j])
            G.add_edge(i, j, weight=weight)

    # Maximum-weight matching
    matching = nx.max_weight_matching(G, maxcardinality=True)

    # Convert to your pair format
    pairs = []
    for i, j in matching:
        pairs.append(((models[i], model_indices[i]),
                      (models[j], model_indices[j])))

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
def random_regular_graph_maxdiff_pairing(tuple_models, val_loader, degree=3, seed=None):
    """
    Option A:
    Include oracle in a true random d-regular graph, then choose disjoint pairs
    greedily by largest accuracy difference among allowed graph edges.

    Returns:
        [((model_a, idx_a), (model_b, idx_b)), ...]
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

    # compute accuracies in tuple_models order
    accs = []
    for model, idx in tuple_models:
        if idx == 0:
            accs.append(100.0)   # keep oracle strongest, same style as your max_difference_pairing
        else:
            accs.append(float(accuracy(model, val_loader)))

    # build a true random d-regular graph
    G = nx.random_regular_graph(d=degree, n=num_models, seed=seed)

    # score only allowed edges
    edge_list = []
    for i, j in G.edges():
        diff = abs(accs[i] - accs[j])
        edge_list.append((diff, i, j))

    # highest-gap first
    edge_list.sort(reverse=True)

    # greedy matching from largest difference edges
    paired = set()
    pairs = []

    for diff, i, j in edge_list:
        if i not in paired and j not in paired:
            paired.add(i)
            paired.add(j)
            pairs.append(((models[i], ids[i]), (models[j], ids[j])))

    return pairs

@torch.no_grad()
def random_regular_graph_uniform_pairing(tuple_models, degree=3, seed=None):
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


def Best_teach_best_pairing(tuple_models, val_loader):

    # Extract models and their initial indices
    models = [model for model, idx in tuple_models]
    initial_indices = [idx for model, idx in tuple_models]

    # Compute test accuracies for all models
    accuracies = torch.tensor([accuracy(model, val_loader) for model in models])
    sorted_indices = torch.argsort(accuracies, descending=True)   # (descending order)
    sorted_models_with_indices = [(models[i], initial_indices[i]) for i in sorted_indices]

    # Step 3: Split into teachers (top half) and remaining models
    num_teachers = len(models) // 2
    teachers_with_indices = sorted_models_with_indices[:num_teachers]
    remaining_with_indices = sorted_models_with_indices[num_teachers:]

    # Pair teachers with remaining models (highest-to-highest)
    pairs = []
    for (teacher, teacher_idx), (student, student_idx) in zip(teachers_with_indices, remaining_with_indices):
        pairs.append(((teacher, teacher_idx), (student, student_idx)))

    return pairs


@torch.no_grad()
def get_output_vectors(model, val_loader):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            #probs = F.softmax(outputs, dim=1)
            #outputs_list.append(probs.cpu())

            log_probs = F.log_softmax(outputs, dim=1)
            outputs_list.append(log_probs.cpu())
    return torch.cat(outputs_list, dim=0).mean(dim=0)  # average prediction vector, get the mean for all the validation samples for each class, output: 1D tensor of shape [num_classes]



@torch.no_grad()
def compute_feature_vector_accuracy(model, val_splits):

    feature_vector = torch.zeros(K)

    for j, subset in enumerate(val_splits):
        if isinstance(subset, torch.utils.data.Dataset):
            generator = torch.Generator() # added to test if the result will be the same
            generator.manual_seed(run_seed)
            loader = DataLoader(subset, generator=generator, batch_size=batch_size, shuffle=False)
        else:
            loader = subset  # already a DataLoader
        #acc = accuracy(model, loader)

        feature_vector[j] = accuracy(model, loader)

    return feature_vector

@torch.no_grad()
def farthest_val_random_split(tuple_models):

    models = [model for model, idx in tuple_models]
    initial_indices = [idx for model, idx in tuple_models]

    # Get the probability vectors (average softmax outputs across val set)
    feature_vectors = [compute_feature_vector_accuracy(model, val_splits) for model in models]
    feature_vectors = torch.stack(feature_vectors)  #stacks the list into a 2D tensor of shape [num_models, num_classes]

    # Compute pairwise distances
    num_models = len(models)
    distance_list = []
    distances = torch.zeros((num_models, num_models)) # Initializes a zero matrix of shape [num_models, num_models], filled with zeros to store the pairwise distances between every pair of models
    for i in range(num_models):
        for j in range(i + 1, num_models):
            dist = torch.norm(feature_vectors[i] - feature_vectors[j], p=2)  # Euclidean distance
            distances[i][j] = distances[j][i] = dist
            distance_list.append((dist.item(), i, j))

    distance_list.sort(reverse=True)

    paired = set()
    pairs = []

    # Select the farthest available pairs greedily
    for dist, i, j in distance_list:
        if i not in paired and j not in paired:
            paired.update([i, j])
            pairs.append(((models[i], initial_indices[i]), (models[j], initial_indices[j])))
        if len(paired) == num_models:
            break
    return pairs



pairing_methods = {
    "split": lambda: farthest_val_random_split(tuple_models=updated_models),
    "ClassDist": lambda: euclidean_distance_class_accuracy(tuple_models=updated_models, val_loader=val_loader, num_classes=num_classes),
    "AccDiff": lambda: max_difference_pairing(tuple_models=updated_models, val_loader=val_loader),
    "MWM_ClassDist": lambda: maximum_weight_matching_class_accuracy(tuple_models=updated_models, val_loader=val_loader, num_classes=num_classes),
    "MWM_AccDiff": lambda: maximum_weight_matching_accuracy_difference(tuple_models=updated_models, val_loader=val_loader),
    
    "Friend_Random": lambda: pair_fixed_groups_random(groups=materialize_groups_from_ids(updated_models, fixed_group_ids), seed=run_seed + round_idx),
    "Friend_MWM_AccDiff": lambda: pair_fixed_groups_mwm_acc_diff(groups=materialize_groups_from_ids(updated_models, fixed_group_ids), val_loader=val_loader),
    "Friend_AccDiff": lambda: pair_fixed_groups_acc_diff(groups=materialize_groups_from_ids(updated_models, fixed_group_ids), val_loader=val_loader),
    
    "RRG_AccDiff": lambda: random_regular_graph_maxdiff_pairing(tuple_models=updated_models, val_loader=val_loader, degree=degree, seed=run_seed + round_idx),
    "RRg_Random": lambda: random_regular_graph_uniform_pairing(tuple_models=updated_models, degree=degree, seed=run_seed + round_idx),

}


