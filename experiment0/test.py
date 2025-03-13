import random

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from experiment0.train import get_data
from experiment0.model0NN import Model0NN

import os


def test_piece_selection_model(model, dataloader, model_name):
    """
    Evaluates the model on the provided dataloader and prints the running batch accuracy
    and the overall accuracy.

    Assumes that the labels are one-hot encoded (shape: [batch_size, 64]).
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            outputs = model(batch_inputs)
            _, preds = torch.max(outputs, dim=1)
            true_labels = batch_labels.argmax(dim=1)
            correct = (preds == true_labels).sum().item()
            total_correct += correct
            total_samples += batch_labels.size(0)
    overall_accuracy = total_correct / total_samples
    print(f"{model_name} accuracy: {overall_accuracy:.4f}")


def test_model(model, dataloader, model_name):
    """
    Evaluates the model on the provided dataloader and prints the overall accuracy.

    Assumes that the labels are multi-hot vectors with exactly two classes labeled as 1.0.
    For each sample, the top 2 predictions (by output logits) are taken as the predicted labels.
    A sample is considered correct if the set of predicted classes exactly matches the set of true classes.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            outputs = model(batch_inputs)
            # Obtain the top 2 predicted class indices for each sample.
            _, preds_top2 = torch.topk(outputs, 2, dim=1)
            # Extract the true indices from the multi-hot encoded labels.
            # Since each label has exactly two ones, topk will return their indices.
            _, true_top2 = torch.topk(batch_labels, 2, dim=1)

            # Sort the indices so that the order does not affect the equality check.
            preds_top2_sorted, _ = torch.sort(preds_top2, dim=1)
            true_top2_sorted, _ = torch.sort(true_top2, dim=1)

            # For each sample, check if both predicted indices match the true indices.
            correct = (preds_top2_sorted == true_top2_sorted).all(dim=1).sum().item()
            total_correct += correct
            total_samples += batch_labels.size(0)

    overall_accuracy = total_correct / total_samples
    print(f"{model_name} accuracy: {overall_accuracy:.4f}")



if __name__ == "__main__":
    random.seed(904056181)
    inputs, outputs = get_data()
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, output_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    for model_path in os.listdir('./models'):
        model = Model0NN.load_model("./models/"+model_path)
        test_model(model, test_loader, model_path)