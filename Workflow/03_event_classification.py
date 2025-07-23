# 03_event_classification.py


import matplotlib
matplotlib.use('Agg')  # <--- Fix: Use Agg for headless environments
import csv
import sys
import os
from pathlib import Path
from datetime import datetime
import platform
import yaml
import git
import pandas as pd
import logging
import configargparse
import tarrow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, RandomSampler
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add tarrow package path dynamically
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "TAP" / "tarrow"))
sys.path.append(str(project_root / "TAP" / "tarrow" / "tarrow"))

# === NEW: Helper functions for class balance/imbalance metrics ===

def gini_index(labels):
    p = np.mean(labels)
    return 2 * p * (1 - p)

def log_normalized_shannon_entropy(labels):
    ps = np.bincount(labels)
    ps = ps / ps.sum()
    ps = ps[ps > 0]
    entropy = -np.sum(ps * np.log(ps))
    max_entropy = np.log(len(np.unique(labels)))
    return entropy / max_entropy if max_entropy > 0 else 0

def imbalance_ratio(labels):
    n_pos = np.sum(labels > 0)
    n_neg = np.sum(labels == 0)
    return n_pos / (n_neg + 1e-8)

def get_class_stats(labels):
    n_total = len(labels)
    n_pos = np.sum(labels > 0)
    n_neg = np.sum(labels == 0)
    return {
        "Total crop pairs": n_total,
        "Positive (class 1)": n_pos,
        "Negative (class 0)": n_neg,
        "Imbalance ratio (pos/neg)": imbalance_ratio(labels),
        "Gini index": gini_index(labels),
        "Log norm. Shannon entropy": log_normalized_shannon_entropy(labels),
    }

def print_and_save_stats(name, labels, outdir, tag):
    stats = get_class_stats(labels)
    print(f"\n{name}:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    # Save to CSV
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"class_balance_{tag}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            writer.writerow([k, v])
    print(f"Saved class balance stats to {csv_path}")

def save_config_metadata(args, output_dir):
    import collections.abc
    def is_basic_type(val):
        return isinstance(val, (str, int, float, bool, type(None))) or (
            isinstance(val, collections.abc.Sequence) and all(is_basic_type(v) for v in val)
        )
    metadata = {
        "timestamp": str(datetime.now().isoformat()),
        "python_version": str(platform.python_version()),
        "torch_version": str(torch.__version__),
        "cuda_available": str(torch.cuda.is_available()),
    }
    try:
        repo = git.Repo(search_parent_directories=True)
        metadata["git_commit"] = str(repo.head.object.hexsha)
    except Exception:
        metadata["git_commit"] = "unknown"

    config_path = Path(output_dir) / "training_config.yaml"
    filtered_args = {k: v for k, v in vars(args).items() if is_basic_type(v)}
    filtered_args = {k: str(v) for k, v in filtered_args.items()}
    with open(config_path, "w") as f:
        yaml.safe_dump({**filtered_args, **metadata}, f)
    print(f"Saved config to {config_path}")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels != out_channels or stride != 1, use a shortcut projection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # Identity function

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, input_shape, num_cls):
        super(SimpleResNet, self).__init__()

        # Unpacking input shape
        batch_size, time_step, channel, height, width = input_shape

        # Initial conv layer takes 32 channels as input
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # One ResNet block, using 64 channels for both input and output to ensure identity connection
        self.block = BasicBlock(64, 64, stride=1)  # No downsampling, identity connection ensured

        # Calculate the flattened size for the FC layer
        self.flattened_size = batch_size*time_step * 32 * height * width  # No downsampling, spatial dimensions unchanged

        # Fully connected layer for classification
        self.fc = nn.Linear(self.flattened_size, num_cls)

    def forward(self, x):
        # Merge the time_step into the batch dimension to handle 2D convolutions
        batch_size, time_step, channel, height, width = x.shape
        x = x.view(batch_size * time_step, channel, height, width)

        # Apply initial convolution and ResNet block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block(x)

        # Reshape for FC layer
        x = x.view(batch_size, time_step, -1)
        x = x.mean(dim=1)  # Temporal mean pooling
        # print(f"Shape after mean pooling: {x.shape}")

        # Classification
        x = self.fc(x)
        return x



def plot_images_gray_scale(image1=None, image2=None, mask1=None, mask2=None):
    """
    plot image1 and image2 side by side for inspections
    :param image1: torch tensor
    :param image2: torch tensor
    :return:
    """
    import matplotlib.pyplot as plt
    if image1 is None:
        image1 = torch.zeros((1,96,96), dtype=torch.uint8)
    if image2 is None:
        image2 = torch.zeros((1,96,96), dtype=torch.uint8)
    if mask1 is None:
        mask1 = torch.zeros((1,96,96), dtype=torch.uint8)
    if mask2 is None:
        mask2 = torch.zeros((1,96,96), dtype=torch.uint8)

    image1_np = image1.squeeze().numpy()
    image2_np = image2.squeeze().numpy()
    mask1_np = mask1.squeeze().numpy()
    mask2_np = mask2.squeeze().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(10, 5))

    axes[0].imshow(image1_np, cmap='gray')
    axes[0].axis('off')  # Turn off axis labels

    axes[1].imshow(image2_np, cmap='gray')
    axes[1].axis('off')

    axes[2].imshow(mask1_np, cmap='gray')
    axes[2].axis('off')

    axes[3].imshow(mask2_np, cmap='gray')
    axes[3].axis('off')

    # Show the plot
    plt.show()


class ClsHead(nn.Module):
    """
    classification head that takes the dense representation from the TAP model as input,
    output raw scores for each class of interest: (nothing of interest, cell division, cell death)
    architecture: fully connected layer [TBD]
    """
    def __init__(self, input_shape, num_cls):
        super(ClsHead, self).__init__()

        batch_size, time_step, channel, height, width = input_shape
        # input shape (Batch, Time, Channel, X, Y)
        self.flattened_size = time_step*channel*height*width

        # using a fully connected layer
        self.fc = nn.Linear(self.flattened_size, num_cls)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# class ClsHead(nn.Module):
#     """
#     time-invariant head
#     Classification head that takes the dense representation from the TAP model as input,
#     and outputs raw scores for each class of interest: (nothing of interest, cell division, cell death).
#     The fully connected layer shares the same weights across the time_step dimension.
#     """
#     def __init__(self, input_shape, num_cls):
#         super(ClsHead, self).__init__()
#
#         batch_size, time_step, channel, height, width = input_shape
#         # input shape (Batch, Time, Channel, X, Y)
#
#         # Create a Conv3d layer with a kernel size of 1 in the time dimension
#         # This makes the weights shared across time steps.
#         self.conv = nn.Conv3d(
#             in_channels=channel,
#             out_channels=num_cls,
#             kernel_size=(2, height, width),  # 1 in time_step, full in spatial dimensions
#             stride=(1, 1, 1),
#             padding=(0, 0, 0)
#         )
#
#     def forward(self, x):
#         # x shape: (Batch, Time, Channel, X, Y)
#         x = x.permute(0, 2, 1, 3, 4)
#         # Apply convolution which shares weights across time steps
#         x = self.conv(x)  # output shape: (Batch, Time, num_cls, 1, 1)
#
#         # # Remove the unnecessary dimensions
#         # x = x.squeeze(-1).squeeze(-1)  # output shape: (Batch, Time, num_cls)
#         #
#         # # Optionally, you can average over the time dimension if you want a single output per batch
#         # x = x.mean(dim=1)  # output shape: (Batch, num_cls)
#
#         x = x.view(x.size(0), -1)
#
#         return x


def reinitialize_weights(model):
    import torch.nn.init as init
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            # Only reinitialize if the weight has 2 or more dimensions
            if len(layer.weight.shape) >= 2:
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                # Handle 1D or fewer dimensions (e.g., BatchNorm, LayerNorm, etc.)
                init.normal_(layer.weight, mean=0.0, std=1.0)
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Reinitialize bias (if it exists)
            init.zeros_(layer.bias)


def train_cls_head(
    train_loader, test_loader, patch_size, num_epochs, random_seed, device,
    model_load_dir, cls_head_arch, TAP_init,
    load_saved_cls_head=False, cls_head_load_path=None,
    output_dir=None
):
    """
    Train the classification head to the task of predicting cell event: no event, division, death
    for the input pair of patches.
    """
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import confusion_matrix, classification_report
    import pandas as pd
    import numpy as np
    import os

    model = tarrow.models.TimeArrowNet.from_folder(model_folder=model_load_dir)
    model.to(device)

    if TAP_init == 'km_uniform':
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        reinitialize_weights(model)
        print(f'- - - Initialising TAP model using {TAP_init} - - - ')
    elif TAP_init == 'loaded':
        print('- - - Initialising TAP model using loaded weights - - - ')

    for parem in model.parameters():
        parem.requires_grad = False

    # shape of the dense representation from the pretrained U-net is '(1, 2, 32, patch_size, patch_size)'
    # fix the random seed for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    if cls_head_arch == 'linear':
        cls_head = ClsHead(input_shape=(1, 2, 32, patch_size, patch_size), num_cls=2).to(device)
    elif cls_head_arch == 'resnet':
        cls_head = SimpleResNet(input_shape=(1, 2, 32, patch_size, patch_size), num_cls=2).to(device)
    elif cls_head_arch == 'minimal':
        # For now, treat 'minimal' as 'linear'
        cls_head = ClsHead(input_shape=(1, 2, 32, patch_size, patch_size), num_cls=2).to(device)
    else:
        raise ValueError(f"Unknown cls_head_arch: {cls_head_arch} (expected 'linear', 'resnet', or 'minimal')")

    if load_saved_cls_head:
        print(f" - - Loading pretrained cls head - - ")
        cls_head_state_dict = torch.load(cls_head_load_path, map_location=device)
        cls_head.load_state_dict(cls_head_state_dict)

    optimizer = optim.Adam(cls_head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        cls_head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for datapoint in train_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            # Forward pass through the pre-trained model to get the dense representation
            with torch.no_grad():
                rep = model.embedding(x)
            # Forward pass through the classification head
            outputs = cls_head(rep)
            # Calculate the loss
            loss = criterion(outputs, y)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # --- TEST LOOP WITH DEBUG PRINTS ---
    running_loss = 0.0
    correct = 0
    total = 0
    count_event_interest = 0
    y_pred = []
    y_true = []
    y_scores = []  # collect softmax probabilities for class 1
    cls_head.eval()
    print("\n=== Test set predictions (DEBUG) ===")
    with torch.no_grad():
        for datapoint in test_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            rep = model.embedding(x)
            outputs = cls_head(rep)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            count_event_interest += (y == 1).sum().item()
            y_pred.extend([t.item() for t in predicted])
            y_true.extend([t.item() for t in y])
            probs = torch.softmax(outputs, dim=1)
            y_scores.extend(probs[:, 1].cpu().numpy())
            # --- DEBUG PRINT for every sample ---
            for i in range(x.size(0)):
                print(f"Sample {i}: True label = {y[i].item()}, "
                      f"Pred = {predicted[i].item()}, "
                      f"Prob_class0 = {probs[i,0].item():.3f}, "
                      f"Prob_class1 = {probs[i,1].item():.3f}")

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    print(f"\nTest Loss: {epoch_loss:.4f}, Test accuracy: {epoch_accuracy:.4f}")
    print(f"There are {count_event_interest} out of {total} crops containing events of interest in the test set")

    cm_test = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm_test, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    print("Confusion Matrix test data:")
    print(cm_df)
    print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))

    # SAVE OUTPUTS FOR VISUALISATION  ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'y_true.npy'), np.array(y_true))
        np.save(os.path.join(output_dir, 'y_pred.npy'), np.array(y_pred))
        np.save(os.path.join(output_dir, 'y_scores.npy'), np.array(y_scores))
        print(f"Saved test predictions to {output_dir}")

    # Distribution of positive labels in training set
    count_event_interest_train = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for datapoint in train_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            count_event_interest_train += (y == 1).sum().item()
            total += y.size(0)
            rep = model.embedding(x)
            outputs = cls_head(rep)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend([t.item() for t in predicted])
            y_true.extend([t.item() for t in y])
    print(f"There are {count_event_interest_train} out of {total} crops containing events of interest in the training set")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    print("Confusion Matrix train data:")
    print(cm_df)
    print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))

    return cls_head, model, cm_test
    

def count_data_points(dataloader):
    count = 0
    num_positive_event = 0
    for batch in dataloader:
        inputs, event_labels, labels = batch[0], batch[1], batch[2]
        count += inputs.size(0)  # Increment by the batch size
        num_positive_event += (event_labels == 1).sum().item()
    return count, num_positive_event


class BalancedSampler(Sampler):
    def __init__(self, data_source, num_crops_per_image, balanced_sample_size, data_gen_seed, sequential=False):
        self.data_source = data_source
        self.sequential = sequential
        self.num_crops_per_image = num_crops_per_image
        self.balanced_sample_size = balanced_sample_size  # the number of samples after resampling and balancing
        self.data_gen_seed = data_gen_seed
        num_image_pairs = len(self.data_source)
        # Separate the initial samples by label
        self.positive_indices = []
        self.negative_indices = []

        for i in range(num_image_pairs):
            if data_source[i][1] > 0:
                self.positive_indices.append(i)
            if data_source[i][1] == 0:
                self.negative_indices.append(i)

        # Ensure equal sampling from both classes
        self.num_samples_per_class = min(self.balanced_sample_size // 2, len(self.positive_indices),
                                         len(self.negative_indices))

    def get_combined_samples(self, data_gen_seed):
        import random
        # for reproducibility
        torch.manual_seed(data_gen_seed)

        if self.sequential:
            positive_samples = self.positive_indices[:self.num_samples_per_class]
            negative_samples = self.negative_indices[:self.num_samples_per_class]
        else:
            positive_samples = torch.multinomial(
                torch.ones(len(self.positive_indices)),
                self.num_samples_per_class,
                replacement=True
            ).tolist()
            positive_samples = [self.positive_indices[i] for i in positive_samples]

            negative_samples = torch.multinomial(
                torch.ones(len(self.negative_indices)),
                self.num_samples_per_class,
                replacement=True
            ).tolist()
            negative_samples = [self.negative_indices[i] for i in negative_samples]

        # Combine positive and negative samples
        # print(f"positive_samples : {len(positive_samples)}, negative_samples : {len(negative_samples)}")
        combined_samples = positive_samples + negative_samples

        # Shuffle the combined samples if not sequential
        if not self.sequential:
            # for reproducibility
            random.seed(data_gen_seed + 123)
            combined_samples = random.sample(combined_samples, len(combined_samples))
        return combined_samples

    def __iter__(self):
        combined_samples = self.get_combined_samples(data_gen_seed=self.data_gen_seed)
        return iter(combined_samples)

    def __len__(self):
        return 2 * self.num_samples_per_class


def probing_mistake_predictions(model, cls_head, test_data_loader, device):
    """
    Output mistake predictions according to the type (e.g. false positive).
    :param model:
    :param test_data:
    :param type_pred_err:
    :param num_outputs:
    :param test_data_loader: batchsize must be 1
    :return:
    """
    false_positives = []
    false_negatives = []
    logits_false_pos = []
    logits_false_neg = []
    cls_head.eval()
    with torch.no_grad():
        for datapoint in test_data_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            # Forward pass through the pre-trained model to get the dense representation
            # Ensure the pre-trained model is not being updated
            rep = model.embedding(x)

            # Forward pass through the classification head
            outputs = cls_head(rep)
            _, predicted = torch.max(outputs, 1)
            datapoint.append(predicted.detach().cpu())
            # datapoint : (x_crop, event_label, label, crop_coordinates, predicted_value)
            # crop_coordinates = (torch.tensor(i), torch.tensor(j), torch.tensor(idx) (time index of the frame), TAP label)
            if (predicted == 1) and (y == 0):
                # false positive
                false_positives.append(datapoint)
                logits_false_pos.append(torch.squeeze(outputs.detach().cpu()))
            elif (predicted == 0) and (y == 1):
                false_negatives.append(datapoint)
                logits_false_neg.append(torch.squeeze(outputs.detach().cpu()))

    return false_positives, false_negatives, logits_false_pos, logits_false_neg


# def save_output_as_txt(data, output_f_path):
#     # Open a file to write the numerical data
#     with open(output_f_path, 'w') as f:
#         for item in data:
#             # Convert the tuple to a list of numbers
#             converted_item = []
#             for element in item:
#                 if isinstance(element, torch.Tensor):
#                     converted_item.append(element.item())  # Extract scalar value from tensor
#                 elif isinstance(element, list):  # Handle list of tensors
#                     converted_item.append([tensor.item() for tensor in element])
#
#             # Write the converted item to the file
#             f.write(str(converted_item) + '\n')
#
#     print(f"Successfully saved to {output_f_path}")


def probing_mistaken_preds(model, cls_head_trained, test_loader_probing, device):
    (false_positives, false_negatives,
     logits_false_pos, logits_false_neg) = probing_mistake_predictions(model,
                                                                       cls_head_trained,
                                                                       test_loader_probing,
                                                                       device)
    false_positives_coordinates = [tuple(e[1:]) for e in false_positives]
    false_negatives_coordinates = [tuple(e[1:]) for e in false_negatives]
    print(f"number of false_positives predictions: {len(false_positives_coordinates)}\n"
          f"number of false_negatives predictions: {len(false_negatives_coordinates)}")
    return (false_positives_coordinates, false_negatives_coordinates,
            false_positives, false_negatives, logits_false_pos, logits_false_neg)


def estimate_total_events(input_data):
    """
    estimate the total number of labels for events using a grid approach. The count is stored in data.
    so this function only aggregates them over the entire sequence of frames.
    :param data:
    :return: total count of event labels for the entire sequence of frames (a.k.a. movie)
    """
    total_count_events = 0
    for i in range(len(input_data)):
        count_current_image = input_data[i][1].detach().item()
        total_count_events += count_current_image
    return total_count_events


def save_as_json(input_data, file_save_path):
    import json
    data_to_save = []
    # Iterate over each item in the original list
    for item in input_data:
        converted_item = []
        # Iterate over each element in the current item
        for element in item:
            # Check if the element is a PyTorch tensor
            if hasattr(element, 'tolist'):
                # Convert the tensor to a list
                converted_item.append(element.tolist())
            else:
                # If it's not a tensor, keep it as is
                converted_item.append(element)

        # Append the converted item to the new list
        data_to_save.append(converted_item)
    # Saving the list as JSON
    with open(file_save_path, 'w') as f:
        json.dump(data_to_save, f)
    print(f"data saved to {file_save_path}")


def data_split(input_image_crops, train_data_ratio, validation_data_ratio, data_seed):
    """
    train, valid, test split. The test split will be determined by train_data_ratio and validation_data_ratio
    as it is given by the rest of the dataset after train and validation data are taken.
    :param input_image_crops:
    :param train_data_ratio:
    :param validation_data_ratio:
    :param data_seed:
    :return:
    """
    import random
    random.seed(data_seed)
    random.shuffle(input_image_crops)

    # Determine split indices
    total_length = len(input_image_crops)
    train_end = int(train_data_ratio * total_length)
    valid_end = train_end + int(validation_data_ratio * total_length)

    # Split the list
    train_data = input_image_crops[:train_end]
    valid_data = input_image_crops[train_end:valid_end]
    test_data = input_image_crops[valid_end:]

    # Verify the sizes
    print(f"Total data points: {total_length}")
    print(f"Training data points: {len(train_data)}")
    print(f"Validation data points: {len(valid_data)}")
    print(f"Test data points: {len(test_data)}")

    return train_data, valid_data, test_data


def multi_runs_training(num_runs, model_seed_init, train_loader, test_loader,
                        size, training_epochs, device, model_load_dir,
                        cls_head_arch,
                        TAP_init,
                        load_saved_cls_head=False,
                        cls_head_load_path=None,
                        output_dir=None):  # <-- ADDED output_dir

    import numpy as np
    import os

    precision_class_0_all = []
    precision_class_1_all = []
    recall_class_0_all = []
    recall_class_1_all = []
    cls_head_trained = None
    model = None

    # These will collect the predictions/scores from the *last run*
    y_true = []
    y_pred = []
    y_scores = []

    for i in range(num_runs):
        model_seed = model_seed_init + i*20
        cls_head_trained, model, cm_test = train_cls_head(
            cls_head_arch=cls_head_arch,
            train_loader=train_loader,
            test_loader=test_loader,
            patch_size=size,
            num_epochs=training_epochs,
            random_seed=model_seed,
            device=device,
            model_load_dir=model_load_dir,
            load_saved_cls_head=load_saved_cls_head,
            cls_head_load_path=cls_head_load_path,
            TAP_init=TAP_init,
            output_dir=output_dir  # <-- PASS output_dir TO SAVE .npy FILES
        )
        precision_class_0 = cm_test[0][0] / (cm_test[0][0] + cm_test[1][0]) if (cm_test[0][0] + cm_test[1][0]) > 0 else float('nan')
        precision_class_1 = cm_test[1][1] / (cm_test[0][1] + cm_test[1][1]) if (cm_test[0][1] + cm_test[1][1]) > 0 else float('nan')
        recall_class_0 = cm_test[0][0] / (cm_test[0][0] + cm_test[0][1]) if (cm_test[0][0] + cm_test[0][1]) > 0 else float('nan')
        recall_class_1 = cm_test[1][1] / (cm_test[1][0] + cm_test[1][1]) if (cm_test[1][0] + cm_test[1][1]) > 0 else float('nan')
        precision_class_0_all.append(precision_class_0)
        precision_class_1_all.append(precision_class_1)
        recall_class_0_all.append(recall_class_0)
        recall_class_1_all.append(recall_class_1)

    # After training, run predictions on the test set for reporting and visualizations
    cls_head_trained.eval()
    with torch.no_grad():
        for datapoint in test_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            rep = model.embedding(x)
            outputs = cls_head_trained(rep)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            y_true.extend([t.item() for t in y])
            y_pred.extend([t.item() for t in predicted])
            y_scores.extend(probs[:, 1].cpu().numpy())

    # --- Save y_true, y_pred, y_scores as .npy files to output_dir (figures) ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'y_true.npy'), np.array(y_true))
        np.save(os.path.join(output_dir, 'y_pred.npy'), np.array(y_pred))
        np.save(os.path.join(output_dir, 'y_scores.npy'), np.array(y_scores))
        print(f"Saved y_true, y_pred, y_scores .npy arrays to {output_dir}")

    return (np.array(precision_class_0_all), np.array(precision_class_1_all),
            np.array(recall_class_0_all), np.array(recall_class_1_all),
            cls_head_trained, model, y_true, y_pred, y_scores)


def save_datasets(train_data_crops_flat, valid_data_crops_flat, test_data_crops_flat, dataset_save_dir):
    import os
    os.makedirs(dataset_save_dir, exist_ok=True)
    torch.save(train_data_crops_flat, os.path.join(dataset_save_dir, 'train_data_crops_flat.pth'))
    torch.save(valid_data_crops_flat, os.path.join(dataset_save_dir, 'valid_data_crops_flat.pth'))
    torch.save(test_data_crops_flat, os.path.join(dataset_save_dir, 'test_data_crops_flat.pth'))
    print(f"Train, validation and test data all saved to {dataset_save_dir}")


class CellEventClassModel(nn.Module):
    def __init__(self, TAPmodel, ClsHead):
        super(CellEventClassModel, self).__init__()
        self._TAPmodel = TAPmodel
        self._ClsHead = ClsHead

    def forward(self, _input):
        z = self._TAPmodel.embedding(_input)
        y = self._ClsHead(z)
        return y

# === CLASSIFICATION VISUALIZER ===
class ClassificationVisualizer:
    def __init__(self, y_true, y_pred, y_scores=None, class_names=None, output_dir="figures"):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_scores = np.array(y_scores) if y_scores is not None else None
        self.class_names = class_names if class_names is not None else ["class 0", "class 1"]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] ClassificationVisualizer will output to: {output_dir}")

    def plot_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        print(f"[DEBUG] Saving confusion_matrix to {os.path.join(self.output_dir, 'confusion_matrix.png')}")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.pdf"))
        np.savetxt(os.path.join(self.output_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
        plt.close()

    def plot_roc_curve(self):
        if self.y_scores is None: return
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        print(f"[DEBUG] Saving roc_curve to {os.path.join(self.output_dir, 'roc_curve.png')}")
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"))
        plt.savefig(os.path.join(self.output_dir, "roc_curve.pdf"))
        plt.close()

    def plot_pr_curve(self):
        if self.y_scores is None: return
        from sklearn.metrics import precision_recall_curve, auc
        import matplotlib.pyplot as plt
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
        ap = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        print(f"[DEBUG] Saving pr_curve to {os.path.join(self.output_dir, 'pr_curve.png')}")
        plt.savefig(os.path.join(self.output_dir, "pr_curve.png"))
        plt.savefig(os.path.join(self.output_dir, "pr_curve.pdf"))
        plt.close()

    def plot_calibration_curve(self):
        if self.y_scores is None: return
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_scores, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
        plt.plot([0, 1], [0, 1], 'k--', label="Perfect")
        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical probability")
        plt.title("Calibration Curve")
        plt.legend()
        plt.tight_layout()
        print(f"[DEBUG] Saving calibration_curve to {os.path.join(self.output_dir, 'calibration_curve.png')}")
        plt.savefig(os.path.join(self.output_dir, "calibration_curve.png"))
        plt.savefig(os.path.join(self.output_dir, "calibration_curve.pdf"))
        plt.close()

    def plot_per_class_metrics(self):
        from sklearn.metrics import classification_report
        import pandas as pd
        import matplotlib.pyplot as plt
        report = classification_report(self.y_true, self.y_pred, target_names=self.class_names, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(self.output_dir, "classification_report.csv"))
        df_metrics = df.loc[self.class_names][['precision', 'recall', 'f1-score']]
        ax = df_metrics.plot(kind='bar', figsize=(7, 4))
        ax.set_ylim(0, 1)
        plt.title("Per-class Metrics")
        plt.ylabel("Score")
        plt.tight_layout()
        print(f"[DEBUG] Saving per_class_metrics to {os.path.join(self.output_dir, 'per_class_metrics.png')}")
        plt.savefig(os.path.join(self.output_dir, "per_class_metrics.png"))
        plt.savefig(os.path.join(self.output_dir, "per_class_metrics.pdf"))
        plt.close()
        df_metrics.to_csv(os.path.join(self.output_dir, "per_class_metrics.csv"))

    def run_all(self):
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_pr_curve()
        self.plot_calibration_curve()
        self.plot_per_class_metrics()

class IoUVisualizer:
    def __init__(self, y_true_masks, y_pred_masks, class_names=None, output_dir="figures"):
        self.y_true_masks = np.array(y_true_masks)
        self.y_pred_masks = np.array(y_pred_masks)
        self.class_names = class_names if class_names is not None else ["class 0", "class 1"]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_iou(self):
        from sklearn.metrics import jaccard_score
        # Assume flattening masks; works for binary and multiclass
        y_true_flat = self.y_true_masks.reshape(len(self.y_true_masks), -1)
        y_pred_flat = self.y_pred_masks.reshape(len(self.y_pred_masks), -1)
        ious = []
        for yt, yp in zip(y_true_flat, y_pred_flat):
            ious.append(jaccard_score(yt, yp, average=None))
        self.ious = np.array(ious)  # shape: (N, n_classes)
        return self.ious

    def plot_iou_histogram(self):
        import matplotlib.pyplot as plt
        iou_flat = self.ious.mean(axis=1)
        plt.figure(figsize=(6, 4))
        plt.hist(iou_flat, bins=20, color="#2386E6", edgecolor="black", alpha=0.7)
        plt.title("IoU Histogram")
        plt.xlabel("IoU")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "iou_histogram.png"))
        plt.savefig(os.path.join(self.output_dir, "iou_histogram.pdf"))
        plt.close()

    def plot_iou_matrix(self):
        # Mean IoU per class
        import matplotlib.pyplot as plt
        import seaborn as sns
        mean_ious = self.ious.mean(axis=0)
        plt.figure(figsize=(6, 2))
        sns.heatmap(mean_ious[np.newaxis, :], annot=True, fmt=".2f",
                    xticklabels=self.class_names, yticklabels=["Mean IoU"], cmap="Blues")
        plt.title("Per-class IoU")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "iou_matrix.png"))
        plt.savefig(os.path.join(self.output_dir, "iou_matrix.pdf"))
        plt.close()

    def save_iou_csv(self):
        import pandas as pd
        df = pd.DataFrame(self.ious, columns=self.class_names)
        df.to_csv(os.path.join(self.output_dir, "per_image_iou.csv"), index=False)

    def run_all(self):
        self.compute_iou()
        self.plot_iou_histogram()
        self.plot_iou_matrix()
        self.save_iou_csv()

def main():
    import time

    p = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )
    
    p.add('--config', is_config_file=True, help='Path to config YAML file')
    p.add_argument("--input_frame")
    p.add_argument("--input_mask")
    p.add_argument("--cam_size", type=int, default=None)
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--n_images")
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--binarize")
    p.add_argument("--timestamp")
    p.add_argument("--backbone", default='unet')
    p.add_argument("--name")
    p.add_argument("--size", type=int, default=96, required=True)
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--batchsize", type=int, default=108)
    p.add_argument("--cam_subsampling", type=int, default=1)
    p.add_argument("--training_epochs", type=int, required=True)
    p.add_argument("--binary_problem", type=bool, default=True)
    p.add_argument('--balanced_sample_size', type=int, default=50000, help="Number of balanced samples per class")
    p.add_argument("--crops_per_image", required=True, type=int)
    p.add_argument("--model_seed", required=True, type=int)
    p.add_argument("--data_seed", required=True, type=int)
    p.add_argument("--data_save_dir", required=True)
    p.add_argument("--num_runs", type=int, required=True)
    p.add_argument("--model_save_dir", required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--load_saved_cls_head", type=bool, default=False)
    p.add_argument("--cls_head_load_path", default=None)
    p.add_argument("--dataset_save_dir", default="runs", help="Directory to save datasets")
    p.add_argument("--TAP_model_load_path")
    p.add_argument("--cls_head_arch")
    p.add_argument("--TAP_init")

    args = p.parse_args()
    if not args.dataset_save_dir:
        args.dataset_save_dir = args.model_save_dir or "runs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    data_load_path = os.path.join(args.data_save_dir, 'preprocessed_image_crops.pth')
    image_crops_flat_loaded = torch.load(data_load_path)
    print(f"image_crops_flat_loaded: {len(image_crops_flat_loaded)}")
    train_data_ratio = 0.6
    validation_data_ratio = 0.2
    train_data_crops_flat, valid_data_crops_flat, test_data_crops_flat = data_split(
        image_crops_flat_loaded,
        train_data_ratio,
        validation_data_ratio,
        args.data_seed
    )
    min_required_crops = 2
    if len(image_crops_flat_loaded) <= min_required_crops:
        print(f"⚠️  Very small dataset ({len(image_crops_flat_loaded)} crops).")
        print("   Using all crops for both training and testing. Splitting skipped.")
        train_data_crops_flat = image_crops_flat_loaded
        valid_data_crops_flat = []
        test_data_crops_flat = image_crops_flat_loaded
    if len(train_data_crops_flat) == 0 or len(test_data_crops_flat) == 0:
        print("\n❌ Not enough data to split into train/test sets!")
        print(f"Train size: {len(train_data_crops_flat)}, Test size: {len(test_data_crops_flat)}")
        print("Please check your input data or use a larger dataset.")
        return

    # === PRINT & SAVE CLASS BALANCE METRICS (before balancing) ===
    vis_outdir = os.path.join(args.data_save_dir, "figures")
    labels_train_before = np.array([x[1] for x in train_data_crops_flat])
    labels_valid = np.array([x[1] for x in valid_data_crops_flat])
    labels_test = np.array([x[1] for x in test_data_crops_flat])

    print_and_save_stats("Train BEFORE balancing", labels_train_before, vis_outdir, "train_before_balancing")
    print_and_save_stats("Validation", labels_valid, vis_outdir, "validation")
    print_and_save_stats("Test", labels_test, vis_outdir, "test")

    estimated_total_event_count = estimate_total_events(image_crops_flat_loaded)
    train_loader = DataLoader(
        train_data_crops_flat,
        sampler=BalancedSampler(
            train_data_crops_flat,
            args.crops_per_image,
            args.balanced_sample_size,
            data_gen_seed=args.data_seed,
            sequential=False
        ),
        batch_size=args.batchsize,
        num_workers=0,
        drop_last=False,
        persistent_workers=False
    )
    torch.manual_seed(args.data_seed)
    test_loader = DataLoader(
        test_data_crops_flat,
        sampler=RandomSampler(test_data_crops_flat),
        batch_size=args.batchsize,
        num_workers=0,
        drop_last=False,
        persistent_workers=False
    )

    # === PRINT & SAVE CLASS BALANCE METRICS (AFTER balancing) ===
    sampler = train_loader.sampler
    indices_balanced = list(sampler.get_combined_samples(args.data_seed))
    labels_balanced = np.array([train_data_crops_flat[i][1] for i in indices_balanced])
    print_and_save_stats("Train AFTER balancing", labels_balanced, vis_outdir, "train_after_balancing")

    print(f"Estimated event count: {estimated_total_event_count}")
    print(f"Train samples and positives: {count_data_points(train_loader)}")
    print(f"Test samples and positives: {count_data_points(test_loader)}")
    print(f"Loading pretrained TAP model from: {args.TAP_model_load_path}")
    test_loader_probing = DataLoader(
        test_data_crops_flat,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        persistent_workers=False
    )
    start_time = time.time()
    print(f"[DEBUG] All figures will be saved in: {vis_outdir}")
    (precision_class_0_all, precision_class_1_all,
     recall_class_0_all, recall_class_1_all,
     cls_head_trained, model, y_true, y_pred, y_scores) = multi_runs_training(
        args.num_runs, args.model_seed, train_loader,
        test_loader, args.size, args.training_epochs,
        device, args.TAP_model_load_path,
        cls_head_arch=args.cls_head_arch,
        TAP_init=args.TAP_init,
        load_saved_cls_head=args.load_saved_cls_head,
        cls_head_load_path=args.cls_head_load_path,
        output_dir=vis_outdir)
    print("Final Metrics (mean, std):")
    for label, values in zip(
        ["Precision Class 0", "Precision Class 1", "Recall Class 0", "Recall Class 1"],
        [precision_class_0_all, precision_class_1_all, recall_class_0_all, recall_class_1_all]
    ):
        print(f"{label}: mean={round(np.mean(values), 2)}, std={round(np.std(values), 2)}")
    end_time = time.time()
    print(f"Time used for model fine-tuning: {round(end_time - start_time, 2)} seconds")

    combined_model = CellEventClassModel(TAPmodel=model, ClsHead=cls_head_trained)
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_save_dir, f'{args.model_id}.pth')
    torch.save(combined_model.state_dict(), model_save_path)
    print(f"Combined model saved to {model_save_path}")
    save_config_metadata(args, args.model_save_dir)
    save_datasets(train_data_crops_flat, valid_data_crops_flat,
                  test_data_crops_flat, args.data_save_dir)
    print("\nGenerating classification visualizations...")
    visualizer = ClassificationVisualizer(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        class_names=["No Event", "Event"],
        output_dir=vis_outdir
    )
    try:
        visualizer.run_all()
    except Exception as e:
        print(f"Error in visualizer: {e}")
    print(f"All classification figures and CSVs saved to {vis_outdir}\n")
    try:
        print("Generating IoU/segmentation visualizations...")
        y_true_masks, y_pred_masks = None, None
        if y_true_masks is not None and y_pred_masks is not None:
            iou_vis = IoUVisualizer(
                y_true_masks=y_true_masks,
                y_pred_masks=y_pred_masks,
                class_names=["Background", "Event"],
                output_dir=vis_outdir
            )
            iou_vis.run_all()
            print(f"All IoU/segmentation figures and CSVs saved to {vis_outdir}\n")
        else:
            print("Skipped IoU visualization: no ground-truth/predicted masks provided.")
    except Exception as e:
        print(f"Error during IoU visualization: {e}")

if __name__ == "__main__":
    main()
