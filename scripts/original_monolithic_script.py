import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torchvision.models as torch_models

import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.utils.data import Subset
from torchvision import datasets, transforms

import time
from datetime import datetime
import pytz

import numpy as np
import os, random
import math

import itertools
import argparse
import matplotlib.pyplot as plt
import networkx as nx

#---------------------------------------------
# Models
#---------------------------------------------

class ResNet(nn.Module):
    def __init__(self, input_channels, pretrained, num_classes):
        super(ResNet, self).__init__()

        # Load a pretrained resnet18 model
        self.base_model = torch_models.resnet18(pretrained=pretrained)

        # Modify the input layer to take grayscale or RGB input
        self.base_model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify the fully connected layer for the number of output classes
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)


    def forward(self, x):
        x = self.base_model(x)
        return x



class LeNet(nn.Module):
    def __init__(self, input_channels, pretrained, num_classes):
        super(LeNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)  # Output: 6x32x32
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Pooling Layer: Output: 6x16x16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Output: 16x12x12

        # Fully Connected Layers
        feature_map_size = 5 if input_channels == 1 else 6

        self.fc1 = nn.Linear(16 * feature_map_size * feature_map_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Activation Function
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        # Initialize weights
        if pretrained:
          self.load_pretrained_weights()


    def forward(self, x):
        x = self.sigmoid(self.conv1(x))  # Convolution 1 + Activation
        x = self.pool(x)                    # Pooling
        x = self.sigmoid(self.conv2(x))  # Convolution 2 + Activation
        x = self.pool(x)                   # Pooling again
        x = self.flatten(x)                 # Flatten
        x = self.sigmoid(self.fc1(x))    # Fully Connected 1 + Activation
        x = self.sigmoid(self.fc2(x))    # Fully Connected 2 + Activation
        x = self.fc3(x)                     # Fully Connected 3 (Output Layer)
        return x

#---------------------------------------------
# variables - settings
#---------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=1, help="Run index for saving results")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# experiments settings
model_class = ResNet
num_models = 10
pairing_strategy = "random_graph_mwm_acc"   # options: "ValSplit_Pair", "Class_Pair"/euclidean, "Acc_Pair"/max, "mwm_classAcc"/maximum-weight matching for class_accuracy distance, "mwm_acc"/maximum-weight matching for accuracy difference, "random_graph"
n_rounds = 100

degree =5

# --- seeds ---
num_runs = args.run_id  

run_seed = 42 # for 1 run
#run_seed = base_seed + 10_000 * run_id # for multiple runs


model_seeds = [
    104729, 1299709, 15485863, 179424673, 2038074743,
    32452843, 49979687, 67867967, 86028121, 104395303,
    122949829, 141650939, 160481183, 179424691, 198491317,
    217645199, 236887691, 256203161, 275604541, 295075153,
]


# global seeding
random.seed(run_seed)
np.random.seed(run_seed)
torch.manual_seed(run_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(run_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)



# train settings
pretrained = False
validation = False

optimizer_type = "sgd"   # options: "sgd", "adam"
optimizers = {}   # reset per run experiment
schedulers = {}   # reset per run experiment

learning_rate = 0.1  # for sgd
momentum = 0.9
weight_decay = 0.0001
use_scheduler = True

gamma = 0.1
temperature = 4
alpha = 0.9


# dataset settings
dataset_name= "cifar_10"  # choose 'mnist', 'fashionmnist', 'cifar_10', or 'cifar_100'
K =10
batch_size = 64
input_channels =3
num_classes = 10
train_size_percent = .80
val_size_percent = .10
test_size_percent = .10

#---------------------------------------------
# Functions
#---------------------------------------------



pairing_methods = {
    "split": lambda: farthest_val_random_split(tuple_models=updated_models),
    "euclidean": lambda: euclidean_distance_class_accuracy(tuple_models=updated_models, val_loader=val_loader, num_classes=num_classes),
    "max": lambda: max_difference_pairing(tuple_models=updated_models, val_loader=val_loader),
    "mwm_classAcc": lambda: maximum_weight_matching_class_accuracy(tuple_models=updated_models, val_loader=val_loader, num_classes=num_classes),
    "mwm_acc": lambda: maximum_weight_matching_accuracy_difference(tuple_models=updated_models, val_loader=val_loader),
    "random_graph": lambda: random_graph_pairing(tuple_models=updated_models, degree=degree, seed=run_seed),
    "random_graph_mwm_acc": lambda: random_graph_mwm_acc_pairing(tuple_models=updated_models, val_loader=val_loader, degree=degree, seed=run_seed + round_idx),
}


def initialize_model(seed: int):
    cpu_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_state = torch.cuda.get_rng_state_all()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = model_class(input_channels, pretrained, num_classes).to(device)

    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(gpu_state)

    return model


@torch.no_grad()
def accuracy(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  return accuracy

@torch.no_grad()
def confidence(model, val_loader):

    model.eval()
    confidences = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1) # Apply softmax to get class probabilities
            max_probs, _ = torch.max(probs, dim=1) # Take the maximum probability

            confidences.append(max_probs)

    avg_confidence = torch.cat(confidences).mean().item()
    return avg_confidence



@torch.no_grad()
def ensemble_accuracy(tuple_models, test_loader, mode="logprob", device=None):
    """
    tuple_models: list of (model, idx) or (model, anything)
    mode: 'prob' (avg probabilities), 'logit' (avg logits), 'logprob' (geometric mean of probs)
    """
    models = [m for m, _ in tuple_models]
    for m in models:
        m.eval()

    if device is None:
        device = next(models[0].parameters()).device

    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        if mode == "logit":
            # Average raw logits
            avg_logits = torch.stack([m(images) for m in models], dim=0).mean(dim=0)  # [B, C]
            preds = avg_logits.argmax(dim=1)

        elif mode == "prob":
            # Average probabilities (arithmetic mean)
            avg_probs = torch.stack([F.softmax(m(images), dim=1) for m in models], dim=0).mean(dim=0)  # [B, C]
            preds = avg_probs.argmax(dim=1)

        else:  # 'logprob' => geometric mean of probabilities (log-opinion pool)
            avg_log_probs = torch.stack([F.log_softmax(m(images), dim=1) for m in models], dim=0).mean(dim=0)  # [B, C]
            # No need to renormalize for argmax; exp preserves argmax
            preds = avg_log_probs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total

@torch.no_grad()    
def average_ensemble_accuracy(tuple_models, test_loader, num_classes):
    device = next(tuple_models[0][0].parameters()).device
    total, correct = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Accumulate log-probabilities from each model
        avg_log_probs = torch.zeros((labels.size(0), num_classes), device=device)

        for model, _ in tuple_models:
            model.eval()
            with torch.no_grad():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                log_probs = torch.log(probs + 1e-12)  # avoid log(0)
                avg_log_probs += log_probs

        # Average the log probabilities
        avg_log_probs /= len(tuple_models)

        # Convert back to normal probabilities for prediction
        avg_probs = torch.exp(avg_log_probs)
        _, predicted = torch.max(avg_probs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

@torch.no_grad()
def average_ensemble_confidence(tuple_models, test_loader, num_classes):
    device = next(tuple_models[0][0].parameters()).device
    total_conf_sum = 0.0
    total_samples = 0

    for images, _ in test_loader:
        images = images.to(device)

        # Accumulate log-probabilities from each model
        avg_log_probs = torch.zeros((images.size(0), num_classes), device=device)

        for model, _ in tuple_models:
            model.eval()
            with torch.no_grad():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                log_probs = torch.log(probs + 1e-12)  # avoid log(0)
                avg_log_probs += log_probs

        # Average the log probabilities across models
        avg_log_probs /= len(tuple_models)

        # Convert back to normal probabilities
        avg_probs = torch.exp(avg_log_probs)  # [B, C]

        # Confidence per sample = max probability among classes
        max_probs = avg_probs.max(dim=1).values  # [B]

        total_conf_sum += max_probs.sum().item()
        total_samples += images.size(0)

    # Return mean confidence (0–1 range)
    return (total_conf_sum / total_samples) if total_samples > 0 else 0.0



@torch.no_grad()
def ensemble_mean_logits(updated_models, images, exclude_idx=None):

    with torch.no_grad():
        logits_list = []
        for m, idx in updated_models:
            if exclude_idx is not None and idx == exclude_idx:
                continue
            logits_list.append(m(images))
        return torch.stack(logits_list, dim=0).mean(dim=0)  # [B, C]



# simple train
def train(model, train_loader, val_loader, num_epochs, use_scheduler, validation):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.55 * num_epochs), int(0.8 * num_epochs)], gamma=gamma)

    for epoch_idx in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if use_scheduler:
            scheduler.step()

        if validation:
            val_accuracy = accuracy(model, val_loader)
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, val_accuracy: {round(val_accuracy, 2)}")

    return model


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
            acc = 1.0   # oracle
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


@torch.no_grad()
def random_graph_pairing(tuple_models, degree=degree, seed=None):
    """
    Pair models under a random-graph connectivity constraint.

    Idea:
    - Each node (model) is allowed to connect only to a few random other nodes
    - Pairing is chosen only from these allowed edges
    - Each model can appear in at most one pair

    Args:
        tuple_models: [((model, idx), ...)] in your usual format
        degree: approximate number of random neighbors per node
        seed: optional seed for reproducibility

    Returns:
        pairs in the same format as your other strategies:
        [((model_a, idx_a), (model_b, idx_b)), ...]
    """
    models = [model for model, _ in tuple_models]
    model_indices = [idx for _, idx in tuple_models]
    num_models = len(tuple_models)

    if num_models < 2:
        return []

    # degree cannot be larger than num_models - 1
    #degree = max(1, min(degree, num_models - 1))

    rng = random.Random(seed) if seed is not None else random

    # Build sparse random graph
    G = nx.Graph()
    G.add_nodes_from(range(num_models))

    # For each node, randomly choose a few neighbors
    for i in range(num_models):
        possible_neighbors = [j for j in range(num_models) if j != i]
        chosen_neighbors = rng.sample(
            possible_neighbors,
            k=min(degree, len(possible_neighbors))
        )

        for j in chosen_neighbors:
            G.add_edge(i, j, weight=1.0)   # unweighted allowed connection

    # Use matching only on allowed edges
    matching = nx.max_weight_matching(G, maxcardinality=True)

    # Convert to your pair format
    pairs = []
    for i, j in matching:
        pairs.append(((models[i], model_indices[i]),
                      (models[j], model_indices[j])))

    return pairs



@torch.no_grad()
def random_graph_mwm_acc_pairing(tuple_models, val_loader, degree=degree, seed=None):
    """
    Pair models using maximum-weight matching on accuracy difference,
    but only over edges allowed by a random graph.

    Args:
        tuple_models: list like [(model, idx), ...]
        val_loader: validation loader for computing model accuracies
        degree: number of random neighbors per node (approximate)
        seed: optional seed for reproducibility

    Returns:
        pairs in format:
        [((model_a, idx_a), (model_b, idx_b)), ...]
    """

    num_models = len(tuple_models)
    if num_models < 2:
        return []

    #degree = max(1, min(degree, num_models - 1))
    rng = random.Random(seed) if seed is not None else random

    models = [m for m, _ in tuple_models]
    ids = [idx for _, idx in tuple_models]

    # ---- step 1: compute accuracy of each model ----
    accs = []
    for model, _ in tuple_models:
        accs.append(float(accuracy(model, val_loader)))

    # ---- step 2: build random sparse graph ----
    G = nx.Graph()
    G.add_nodes_from(range(num_models))

    for i in range(num_models):
        possible_neighbors = [j for j in range(num_models) if j != i]
        chosen_neighbors = rng.sample(
            possible_neighbors,
            k=min(degree, len(possible_neighbors))
        )

        for j in chosen_neighbors:
            if not G.has_edge(i, j):
                weight = abs(accs[i] - accs[j])   # mwm_acc weight
                G.add_edge(i, j, weight=weight)

    # ---- step 3: maximum-weight matching on allowed edges only ----
    matching = nx.max_weight_matching(G, maxcardinality=True)

    # ---- step 4: convert to your pair format ----
    pairs = []
    for i, j in matching:
        pairs.append(
            ((models[i], ids[i]), (models[j], ids[j]))
        )

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



#-----------------------------------------------

def compute_average_accuracy(tuple_models, test_loader):

    total_accuracy = 0.0
    # Compute accuracy for each model
    for model, _ in tuple_models:
        total_accuracy += accuracy(model, test_loader)

    # Calculate average accuracy
    average_accuracy = total_accuracy / len(tuple_models)
    return average_accuracy


def calculate_std_dev(values):
    if len(values) < 2:
        return 0  # Standard deviation is not defined for less than 2 values
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)




class TimeLogger:
    def __init__(self, timezone='US/Eastern'):
        self.timezone = pytz.timezone(timezone)

    def __enter__(self):
        self.start_time = datetime.now(pytz.utc).astimezone(self.timezone)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = datetime.now(pytz.utc).astimezone(self.timezone)
        print(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = self.end_time - self.start_time
        print(f"Duration: {duration}")



@torch.no_grad()
def variance_model_accuracy(models_with_idx, dataloader):

    accs = torch.tensor([accuracy(m, dataloader) for m, _ in models_with_idx], dtype=torch.float32)
    return float(accs.var(unbiased=False).item())


#---------------------------------------------
# Data
#---------------------------------------------

def load_dataset(dataset_name, batch_size):

    # Dataset transformations
    if dataset_name.lower() in ["mnist", "fashionmnist"]:
        transform = transforms.Compose([transforms.ToTensor()])
    else:  # CIFAR datasets
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Load dataset
    if dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset_name.lower() == "fashionmnist":
        dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset_name.lower() == "cifar_10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataset_name.lower() == "cifar_100":
        dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please choose 'mnist', 'fashionmnist', 'cifar_10', or 'cifar_100'.")

    # Create a DataLoader for the full dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, data_loader



def split_dataset(dataset, batch_size, train_size_percent, val_size_percent, split_seed: int):

    train_size = int(train_size_percent * len(dataset))
    val_size = int(val_size_percent * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # seed the split
    split_gen = torch.Generator()
    split_gen.manual_seed(split_seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=split_gen
    )

    # seed the loader order
    loader_gen = torch.Generator()
    loader_gen.manual_seed(split_seed)

    train_loader = DataLoader(train_dataset, generator=loader_gen, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   generator=loader_gen, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  generator=loader_gen, batch_size=batch_size, shuffle=False)

    val_splits = split_validation_set_kfold(val_dataset, K, split_seed)
    return train_loader, val_loader, test_loader, val_splits



def split_validation_set_kfold(dataset, K: int, seed: int):
    length = len(dataset)
    lengths = [length // K] * K
    for i in range(length % K):
        lengths[i] += 1

    g = torch.Generator().manual_seed(seed)
    splits = random_split(dataset, lengths, generator=g)
    return splits



def seed_worker(worker_id: int):
    # Ensures each dataloader worker has a deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cifar_dataset(batch_size: int, seed: int = 42, num_workers: int = 4, root: str = "./data"):
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Two separate dataset objects (same underlying CIFAR files, different transforms)
    full_train_aug  = datasets.CIFAR10(root=root, train=True,  download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=root, train=True,  download=True, transform=test_transform)

    train_size = int(0.9 * len(full_train_aug))
    val_size   = len(full_train_aug) - train_size

    g_split = torch.Generator().manual_seed(seed)
    train_subset_aug, val_subset_indices = random_split(
        range(len(full_train_aug)), [train_size, val_size], generator=g_split
    )

    # Build Subsets using the *same indices* but different dataset objects
    train_dataset = torch.utils.data.Subset(full_train_aug,  train_subset_aug.indices)
    val_dataset   = torch.utils.data.Subset(full_train_eval, val_subset_indices.indices)

    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    # Dataloader RNG (controls shuffle deterministically)
    g_loader = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g_loader, persistent_workers=(num_workers > 0), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0), pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0), pin_memory=True
    )

    return train_loader, val_loader, test_loader



def train_student_oracle(student, train_loader, optimizer, lr_scheduler=None):
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

def train_student_kd(student, teacher, train_loader, optimizer, lr_scheduler=None):
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

@torch.no_grad()
def get_models_no_oracle(updated_models, oracle_id=0):
    return [(m, mid) for (m, mid) in updated_models if mid != oracle_id]


@torch.no_grad()
def ensemble_val_acc_no_oracle(updated_models, val_loader, num_classes, oracle_id=0):
    models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
    return average_ensemble_accuracy(models_no_oracle, val_loader, num_classes)


@torch.no_grad()
def pairwise_disagreement_matrix(models_with_idx, loader, num_classes, device=None):
    ids, preds, _ = collect_preds_and_probs(models_with_idx, loader, num_classes, device=device)
    M, N = preds.shape
    D = torch.zeros((M, M), dtype=torch.float32)

    for i in range(M):
        for j in range(i+1, M):
            dij = (preds[i] != preds[j]).float().mean()
            D[i, j] = D[j, i] = dij

    mean_pairwise = D.sum() / (M*(M-1))  # average over ordered pairs, excludes diagonal automatically
    return ids, D, float(mean_pairwise.item())

def distill_ensemble_to_student(
    ensemble_models,
    student_model,
    train_loader,
    val_loader,
    device,
    temperature=4.0,
    alpha=0.7,
    epochs=10,
    lr=0.01
):

    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)
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


#############################################################################################################



# 1. load original dataset
train_loader, val_loader, test_loader = cifar_dataset(batch_size=batch_size, seed=run_seed, num_workers=4)



# 2. create models
assert num_models <= len(model_seeds), "Not enough seeds!"

# Initialize all models (idx=0 is oracle automatically)
init_models = [ initialize_model(seed=s) for s in model_seeds[:num_models]]

# Attach indices
updated_models = [(model, idx) for idx, model in enumerate(init_models)]
oracle_id = 0  # model-id reserved for Oracle (uses true labels)


# 3. create empty tensors and tuple of models
avg_val_acc_tensor = torch.zeros(n_rounds + 1, device=device)
avg_val_conf_tensor = torch.zeros(n_rounds + 1, device=device)


# direct student-learning progress trackers
# Oracle: one trained student per round -> direct pre/post val-accuracy gain of that student
# Non-oracle: multiple trained students per round -> sum of direct pre/post val-accuracy gains
oracle_gain_student = torch.zeros(n_rounds, dtype=torch.float32)
oracle_gain_student_cum = torch.zeros(n_rounds + 1, dtype=torch.float32)

non_oracle_gain_mean = torch.zeros(n_rounds, dtype=torch.float32)
non_oracle_gain_mean_cum = torch.zeros(n_rounds + 1, dtype=torch.float32)

oracle_gain_ens = torch.zeros(n_rounds, dtype=torch.float32)
oracle_gain_ens_cum = torch.zeros(n_rounds + 1, dtype=torch.float32)

non_oracle_gain_ens = torch.zeros(n_rounds, dtype=torch.float32)
non_oracle_gain_ens_cum = torch.zeros(n_rounds + 1, dtype=torch.float32)






# 4. Training 
with TimeLogger():

    # persistent optimizer/scheduler per model id (so momentum etc. persists across rounds)
    optimizers = {}
    schedulers = {}
    
    avg_val_acc_tensor[0] = average_ensemble_accuracy(updated_models, val_loader, num_classes)
    avg_val_conf_tensor[0] = average_ensemble_confidence(updated_models, val_loader, num_classes)
    
    for round_idx in range(n_rounds):

        if pairing_strategy not in pairing_methods:
            raise ValueError(f"Invalid pairing strategy '{pairing_strategy}'")
        
        print(f"\n=== Round {round_idx + 1}/{n_rounds} ({pairing_strategy}) ===")

        round_start_acc = ensemble_val_acc_no_oracle(updated_models, val_loader, num_classes, oracle_id)
        oracle_phase_acc = round_start_acc
        #oracle_done_this_round = False

        # 1) pair models for this round
        pairs = pairing_methods[pairing_strategy]()
        
        round_kd_total_gain = 0.0
        round_kd_student_count = 0
        
        # 2) train student in each pair
        for (m1, id1), (m2, id2) in pairs:
            
            # Case A: one side is Oracle -> supervised CE on true labels for the other model
            if id1 == oracle_id or id2 == oracle_id:
                student, student_id = (m2, id2) if id1 == oracle_id else (m1, id1)
                
                stu_acc_before_oracle = accuracy(student, val_loader)
                
                opt, sch = get_opt_sched(student_id, student)
                print(f"  oracle-train: oracle -> student {student_id}")

                student = train_student_oracle(
                    student=student,
                    train_loader=train_loader,
                    optimizer=opt,
                    lr_scheduler=sch,
                )
                
                # write back update
                for i, (m, mid) in enumerate(updated_models):
                    if mid == student_id:
                        updated_models[i] = (student, student_id)
                        break

                #oracle_phase_acc = ensemble_val_acc_no_oracle(updated_models, val_loader, num_classes, oracle_id)
                
                #oracle_gain_ens_var = oracle_phase_acc - round_start_acc
                #oracle_gain_ens[round_idx] = oracle_gain_ens_var
                #oracle_gain_ens_cum[round_idx + 1] = oracle_gain_ens_cum[round_idx] + oracle_gain_ens_var
                
                oracle_gain_student_var = (accuracy(student, val_loader)) - stu_acc_before_oracle
                oracle_gain_student[round_idx] = oracle_gain_student_var
                oracle_gain_student_cum[round_idx + 1] = oracle_gain_student_cum[round_idx] + oracle_gain_student_var

                continue

            # Case B: peer-to-peer -> pick mentor (teacher) by current val accuracy, then KD
            acc1 = float(accuracy(m1, val_loader))
            acc2 = float(accuracy(m2, val_loader))

            if acc2 > acc1:
                teacher, teacher_id, teacher_acc = m2, id2, acc2
                student, student_id, student_acc = m1, id1, acc1
            else:
                teacher, teacher_id, teacher_acc = m1, id1, acc1
                student, student_id, student_acc = m2, id2, acc2

            opt, sch = get_opt_sched(student_id, student)

            print(f"  kd-train: teacher {teacher_id} ({teacher_acc:.2f}) -> student {student_id} ({student_acc:.2f})")

            student_acc_before = student_acc
            
            student = train_student_kd(
                student=student,
                teacher=teacher,
                train_loader=train_loader,
                optimizer=opt,
                lr_scheduler=sch,
            )
            
            student_acc_after = float(accuracy(student, val_loader))
            kd_gain = student_acc_after - student_acc_before
            
            # accumulate round statistics
            round_kd_total_gain += kd_gain
            round_kd_student_count += 1
            
            # write back update
            for i, (m, mid) in enumerate(updated_models):
                if mid == student_id:
                    updated_models[i] = (student, student_id)
                    break

        # 3) after finishing the round, log ensemble stats
        models_no_oracle = get_models_no_oracle(updated_models, oracle_id)
        round_end_acc = average_ensemble_accuracy(models_no_oracle, val_loader, num_classes)
        
        avg_val_acc_tensor[round_idx + 1] = average_ensemble_accuracy(models_no_oracle, val_loader, num_classes)
        avg_val_conf_tensor[round_idx + 1] = average_ensemble_confidence(models_no_oracle, val_loader, num_classes)

        if round_kd_student_count > 0:
            round_kd_mean_gain = round_kd_total_gain / round_kd_student_count
        else:
            round_kd_mean_gain = 0.0
        
        non_oracle_gain_mean[round_idx] = round_kd_mean_gain
        non_oracle_gain_mean_cum[round_idx + 1] = non_oracle_gain_mean_cum[round_idx] + round_kd_mean_gain
        
        #base_for_non_oracle = oracle_phase_acc if oracle_done_this_round else round_start_acc
        #non_oracle_gain_ens_var = round_end_acc - oracle_phase_acc
        #non_oracle_gain_ens[round_idx] = non_oracle_gain_ens_var
        #non_oracle_gain_ens_cum[round_idx + 1] = non_oracle_gain_ens_cum[round_idx] + non_oracle_gain_ens_var

        #print(f"  mean non-oracle gain per student = {round_kd_mean_gain:+.2f}")
        #print(f"  round oracle gain      = {float(oracle_gain_per_round_student[round_idx]):.4f}")
        #print(f"  round non-oracle gain ensemble = {float(non_oracle_gain_per_round[round_idx]):.4f}")
        #print(f"  cum oracle gain        = {float(oracle_gain_cum[round_idx + 1]):.4f}")
        #print(f"  cum non-oracle gain    = {float(non_oracle_gain_cum[round_idx + 1]):.4f}")

    time.sleep(5)


rounds = np.arange(1, n_rounds + 1)
rounds_cum = np.arange(1, n_rounds + 2)

plt.figure(figsize=(9, 5))
plt.plot( rounds, oracle_gain_student.cpu().numpy(), label='Oracle Gain student', linewidth=2)
plt.plot( rounds, non_oracle_gain_mean.cpu().numpy(), label='Non-Oracle Gain Mean per Model', linewidth=2)
plt.xlabel("Round")
plt.ylabel("Accuracy Gain")
plt.title("Oracle vs Non-Oracle Gain per Round")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig( f"{num_models}_Mean_{model_class.__name__.lower()}_{pairing_strategy}_r{num_runs}.png", dpi=200)
plt.show()


plt.figure(figsize=(9, 5))
plt.plot( rounds_cum, oracle_gain_student_cum.cpu().numpy(), label='Cumulative Oracle Gain student', linewidth=2)
plt.plot( rounds_cum, non_oracle_gain_mean_cum.cpu().numpy(), label='Cumulative Non-Oracle Gain Mean per Model', linewidth=2)
plt.xlabel("Round")
plt.ylabel("Cumulative Accuracy Gain")
plt.title("Aggregate Progress by Session Type")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig( f"{num_models}_cumulative_mean__{model_class.__name__.lower()}_{pairing_strategy}_r{num_runs}.png", dpi=200)
plt.show()





# Derive model name automatically
model_name = model_class.__name__.lower()   


# save tensors
#torch.save(oracle_gain_student, f"{num_models}_oracle_tensor_{model_name}_{pairing_strategy}_r{num_runs}")
#torch.save(non_oracle_gain_mean, f"{num_models}_non_oracle_tensor_{model_name}_{pairing_strategy}_r{num_runs}") 
torch.save(oracle_gain_student_cum, f"{num_models}_oracle_cum_tensor_{model_name}_{pairing_strategy}_r{num_runs}")
torch.save(non_oracle_gain_mean_cum, f"{num_models}_non_oracle_cum_tensor_{model_name}_{pairing_strategy}_r{num_runs}")


# save tensors
#torch.save(avg_val_acc_tensor, f"avg_ens_acc_tensor_{num_models}_{model_name}_{pairing_strategy}_r{num_runs}")
#torch.save(avg_val_conf_tensor, f"avg_ens_conf_tensor_{num_models}_{model_name}_{pairing_strategy}_r{num_runs}") 

# Save models with their indices
#torch.save([(model.state_dict(), idx) for model, idx in updated_models], f"trained_models_{num_models}_{model_name}_{pairing_strategy}_KD_r{num_runs}")

#=============
#print("ensemble_acc, ensemble_acc_prob, ensemble_acc_logit, avg_acc, avg_conf:")

# Evaluate ensemble performance Ensemble (logprob) Top-1 Accuracy: 
#ensemble_acc = ensemble_accuracy(models_no_oracle, test_loader, mode='logprob')
#print(f"{ensemble_acc:.2f}")

# Optionally compare with averaging probabilities Ensemble (prob) Top-1 Accuracy: 
#ensemble_acc_prob = ensemble_accuracy(models_no_oracle, test_loader, mode='prob')
#print(f"{ensemble_acc_prob:.2f}")

# Or averaging logits Ensemble (logit) Top-1 Accuracy: 
#ensemble_acc_logit = ensemble_accuracy(models_no_oracle, test_loader, mode='logit')
#print(f"{ensemble_acc_logit:.2f}")

# Compute average prediction Average Ensemble Test Accuracy = 
#models = [(m, idx) for idx, m in enumerate(init_models)]
avg_acc = average_ensemble_accuracy(models_no_oracle, test_loader, num_classes)
print(f"avg_ens_acc = {round(avg_acc, 2)}")

# Average Ensemble Test Confidence = 
avg_conf = average_ensemble_confidence(models_no_oracle, test_loader, num_classes)
print(f"avg_ens_conf = {round(avg_conf, 2)}")


ids, D, mean_dis = pairwise_disagreement_matrix(models_no_oracle, val_loader, num_classes)
print("Mean pairwise 0/1 disagreement:", round(mean_dis, 4))

#=============

print("\n===== Distilling Ensemble → Single Model =====")

student_model = initialize_model(seed=999).to(device)

student_model = distill_ensemble_to_student(
    ensemble_models=models_no_oracle,
    student_model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    temperature=4.0,
    alpha=0.7,
    epochs=50,
    lr=0.01
)

student_test_acc = accuracy(student_model, test_loader)

print(f"\nDistilled Student Test Accuracy: {student_test_acc:.2f}")




#=============

# compute each models accuracy 
accuracies = []
for model, idx in models_no_oracle:  # Unpack the model and idx
    acc = accuracy(model, test_loader)
    print(f"M{idx}_acc= {round(acc, 2)}")
    accuracies.append(acc)
    
    
# Standard deviation
std_dev = calculate_std_dev(accuracies)
print(f"std_dev= {round(std_dev, 2)}")



acc_var_test = float(np.var(accuracies))
print(f"acc_var= {round(acc_var_test, 4)}")








    
    

