# Taken from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
import torch
from torch import nn, optim
from torch.nn import functional as F

import sys, os
custom_dir = '/home/cangxiong/projects/synergy_project/TAP/tarrow/'
sys.path.insert(0, custom_dir)
sys.path.append('/home/cangxiong/projects/synergy_project/TAP/tarrow/tarrow')
# print("sys.path:", sys.path)
import tarrow
from torch.utils.data import ConcatDataset, Subset, Dataset, Sampler, DataLoader, RandomSampler, SequentialSampler
from pdb import set_trace


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


# class ClsHead(nn.Module):
#     def __init__(self, input_shape, num_cls):
#         super(ClsHead, self).__init__()
#
#         # Unpacking input shape
#         batch_size, time_step, channel, height, width = input_shape
#
#         # Initial conv layer takes 32 channels as input
#         self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         # One ResNet block, using 64 channels for both input and output to ensure identity connection
#         self.block = BasicBlock(64, 64, stride=1)  # No downsampling, identity connection ensured
#
#         # Calculate the flattened size for the FC layer
#         self.flattened_size = batch_size*time_step * 32 * height * width  # No downsampling, spatial dimensions unchanged
#
#         # Fully connected layer for classification
#         self.fc = nn.Linear(self.flattened_size, num_cls)
#
#     def forward(self, x):
#         # Merge the time_step into the batch dimension to handle 2D convolutions
#         batch_size, time_step, channel, height, width = x.shape
#         x = x.view(batch_size * time_step, channel, height, width)
#
#         # Apply initial convolution and ResNet block
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.block(x)
#
#         # Reshape for FC layer
#         x = x.view(batch_size, time_step, -1)
#         x = x.mean(dim=1)  # Temporal mean pooling
#         # print(f"Shape after mean pooling: {x.shape}")
#
#         # Classification
#         x = self.fc(x)
#         return x


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


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self._device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits, self.temperature)

    def temperature_scale(self, logits, temperature):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = temperature.to(self._device)
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # self.cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for datapoint in valid_loader:
                input, label = datapoint[0].to(self._device), datapoint[1].to(self._device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            # self.temperature.to(self._device)
            loss = nll_criterion(self.temperature_scale(logits, self.temperature), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits, self.temperature), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits, self.temperature), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class CellEventClassModel(nn.Module):
    def __init__(self, TAPmodel, ClsHead):
        super(CellEventClassModel, self).__init__()
        self._TAPmodel = TAPmodel
        self._ClsHead = ClsHead
        # self._device = device

    def forward(self, _input):
        z = self._TAPmodel.embedding(_input)
        y = self._ClsHead(z)
        return y


def count_data_points(dataloader):
    count = 0
    num_positive_event = 0
    for batch in dataloader:
        inputs, event_labels, labels = batch[0], batch[1], batch[2]
        count += inputs.size(0)  # Increment by the batch size
        num_positive_event += (event_labels == 1).sum().item()
    return count, num_positive_event


def probing_mistaken_preds(model, test_loader, device):
    from torch.nn import Softmax, Sigmoid
    false_positives = []
    false_negatives = []
    logits_false_pos = []
    logits_false_neg = []
    all_preds_with_label = []
    model.eval() # remove this as model does not seem to have the eval attribute
    with torch.no_grad():
        for datapoint in test_loader:
            x, y = datapoint[0].to(device), datapoint[1].to(device)
            # Forward pass through the pre-trained model to get the dense representation
            # Ensure the pre-trained model is not being updated
            outputs = model(x)
            # rep = model.embedding(x)

            # Forward pass through the classification head
            # outputs = cls_head_trained(rep)
            _, predicted = torch.max(outputs, 1)
            datapoint.append(predicted.detach().cpu())
            # all_preds_with_label.append((Sigmoid()(outputs.squeeze())[1].detach().cpu(),
            #                              y.detach().cpu(), predicted.detach().cpu()))
            all_preds_with_label.append((Sigmoid()(outputs.squeeze())[1].detach().cpu(),
                                         y.detach().cpu(), predicted.detach().cpu()))
            # datapoint : (x_crop, event_label, label, crop_coordinates, predicted_value)
            # crop_coordinates = (torch.tensor(i), torch.tensor(j), torch.tensor(idx) (time index of the frame), TAP label)
            if (predicted == 1) and (y == 0):
                # false positive
                false_positives.append(datapoint)
                logits_false_pos.append(torch.squeeze(outputs.detach().cpu()))
            elif (predicted == 0) and (y == 1):
                false_negatives.append(datapoint)
                logits_false_neg.append(torch.squeeze(outputs.detach().cpu()))

    false_positives_coordinates = [tuple(e[1:]) for e in false_positives]
    false_negatives_coordinates = [tuple(e[1:]) for e in false_negatives]
    print(f"number of false_positives predictions: {len(false_positives_coordinates)}\n"
          f"number of false_negatives predictions: {len(false_negatives_coordinates)}")
    return (false_positives_coordinates, false_negatives_coordinates,
            false_positives, false_negatives, logits_false_pos, logits_false_neg, all_preds_with_label)


def save_output_as_txt(data, output_f_path):
    # Open a file to write the numerical data
    with open(output_f_path, 'w') as f:
        for item in data:
            # Convert the tuple to a list of numbers
            converted_item = []
            # if item is a single float number
            if isinstance(item, float):
                converted_item.append(str(item))
            else:
                for element in item:
                    if isinstance(element, torch.Tensor):
                        converted_item.append(str(element.item()))  # Extract scalar value from tensor
                    elif isinstance(element, list):  # Handle list of tensors
                        for tensor in element:
                            converted_item.append(str(tensor.item()))
            line = ','.join(converted_item)
            # Write the converted item to the file
            f.write(line + '\n')

    print(f"Successfully saved to {output_f_path}")


def find_interval_index(x, sequence):
    for i in range(len(sequence) - 1):
        if sequence[i] <= x < sequence[i + 1]:
            return i
    # If the value is exactly 1, it belongs to the last interval
    if x == sequence[-1]:
        return len(sequence) - 2
    return None  # In case the value doesn't fall within the expected range


def save_ouput_as_json(input_dict, save_path):
    import json
    with open(save_path, 'w') as json_file:
        json.dump(input_dict, json_file, indent=4)  # 'indent' makes the file more readable
    print(f'Input saved as JSON!')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--TAP_model_load_path")
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--combined_model_load_dir")
    parser.add_argument("--model_id")
    parser.add_argument("--validation_data_path")
    parser.add_argument("--test_data_load_path")
    parser.add_argument("--mistake_pred_dir")
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--cls_head_arch", help='linear or resnet')
    args = parser.parse_args()
    devicestring = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(devicestring)
    print("Running on "+devicestring)
    print(f" - - - loading pretrained TAP model from : {args.TAP_model_load_path} - - - ")

    TAPmodel = tarrow.models.TimeArrowNet.from_folder(model_folder=args.TAP_model_load_path)
    TAPmodel.to(device)
    if args.cls_head_arch == 'linear':
        cls_head = ClsHead(input_shape=(1, 2, 32, args.patch_size, args.patch_size), num_cls=2).to(device)
    elif args.cls_head_arch == 'resnet':
        cls_head = SimpleResNet(input_shape=(1, 2, 32, args.patch_size, args.patch_size), num_cls=2).to(device)
    # cls_head = ClsHead(input_shape=(1, 2, 32, args.patch_size, args.patch_size), num_cls=2).to(device)
    event_rec_model = CellEventClassModel(TAPmodel=TAPmodel, ClsHead=cls_head)
    event_rec_model_state_dict = torch.load(os.path.join(args.combined_model_load_dir, f'{args.model_id}.pth'))
    event_rec_model.load_state_dict(event_rec_model_state_dict)
    print(f" - - Loaded pretrained model {args.model_id} ")
    event_rec_model.to(device)
    for parem in event_rec_model.parameters():
        parem.requires_grad = False

    # valid_data_crops_flat used here
    valid_data_crops_flat = torch.load(args.validation_data_path)
    valid_loader = DataLoader(
        valid_data_crops_flat,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        persistent_workers=False
    )
    # Create a DataLoader from the SAME VALIDATION SET used to train orig_model
    print(f"number of data points in valid_loader, : {count_data_points(valid_loader)}")

    test_data_crops_flat = torch.load(args.test_data_load_path)
    test_loader = DataLoader(
        test_data_crops_flat,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        persistent_workers=False
    )
    print(f"number of data points in test_loader, : {count_data_points(test_loader)}")

    (false_positive_coordinates_orig, false_negative_coordinates_orig,
     false_positive_egs_orig, false_negative_egs_orig,
     logits_false_pos_orig, logits_false_neg_orig, all_preds_with_label_orig) = probing_mistaken_preds(event_rec_model,
                                                                                                       test_loader,
                                                                                                       device)
    scaled_model = ModelWithTemperature(event_rec_model, device=device)
    scaled_model.set_temperature(valid_loader)
    print(f' - - - after temperature scaling - - - ')
    (false_positive_coordinates, false_negative_coordinates,
     false_positive_egs, false_negative_egs,
     logits_false_pos, logits_false_neg, all_preds_with_label_scaled) = probing_mistaken_preds(scaled_model,
                                                                                               test_loader,
                                                                                               device)

    def bin_by_confidence(num_bins, preds_with_label):
        import numpy as np
        """
        return a list of avg accuracy at each bin
        :param num_bins:
        :param preds_with_label: Each element is a tuple ((Sigmoid()(outputs.squeeze())[1].detach().cpu(), 
        y.detach().cpu(), predicted.detach().cpu()))
        :return:
        """
        bins = np.linspace(0, 1, num_bins+1)
        accuracy_per_bin = {}
        num_points_per_bin = {}
        # initialise a list which will be used to store the mean accuracy at each confidence level
        avg_acc_per_bin = [0 for i in range(num_bins)]
        for k in range(num_bins):
            accuracy_per_bin[k] = []
        for i in range(len(preds_with_label)):
            bin_index = find_interval_index(preds_with_label[i][0], bins)
            # in binary case, only consider accuracy and confidence for class 1
            _if_accurate = int((preds_with_label[i][2] == 1) and (preds_with_label[i][1] == 1))
            accuracy_per_bin[bin_index].append(_if_accurate)
        for k in range(num_bins):
            avg_acc_per_bin[k] = np.mean(accuracy_per_bin[k])
            num_points_per_bin[k] = len(accuracy_per_bin[k])
        return bins, avg_acc_per_bin, num_points_per_bin

    bins, avg_acc_per_bin_orig, num_points_per_bin_orig = bin_by_confidence(num_bins=args.num_bins,
                                                   preds_with_label=all_preds_with_label_orig)
    bins, avg_acc_per_bin_scaled, num_points_per_bin_scaled = bin_by_confidence(num_bins=args.num_bins,
                                                     preds_with_label=all_preds_with_label_scaled)
    print(f'bins, avg_acc_per_bin_orig, avg_acc_per_bin_scaled {bins}, {avg_acc_per_bin_orig}, {avg_acc_per_bin_scaled}')

    mistake_pred_model_id_dir = os.path.join(args.mistake_pred_dir, f'calibrated_{args.model_id}')

    os.makedirs(mistake_pred_model_id_dir, exist_ok=True)
    save_output_as_txt(logits_false_pos, os.path.join(mistake_pred_model_id_dir, 'false_positives_logits.txt'))
    save_output_as_txt(logits_false_neg, os.path.join(mistake_pred_model_id_dir, 'false_negatives_logits.txt'))
    print(f"false positive and false negatives examples saved to {mistake_pred_model_id_dir}")

    save_output_as_txt(avg_acc_per_bin_orig, os.path.join(mistake_pred_model_id_dir, 'avg_acc_per_bin_orig.txt'))
    save_output_as_txt(avg_acc_per_bin_scaled, os.path.join(mistake_pred_model_id_dir, 'avg_acc_per_bin_scaled.txt'))
    save_output_as_txt(bins, os.path.join(mistake_pred_model_id_dir, 'confidence_bins.txt'))
    print(f"avg_acc_per_bin_orig, avg_acc_per_bin_scaled saved to {mistake_pred_model_id_dir}")

    save_ouput_as_json(num_points_per_bin_orig, os.path.join(mistake_pred_model_id_dir, 'num_points_per_bin_orig.json'))
    save_ouput_as_json(num_points_per_bin_scaled, os.path.join(mistake_pred_model_id_dir, 'num_points_per_bin_scaled.json'))


if __name__ == '__main__':
    main()