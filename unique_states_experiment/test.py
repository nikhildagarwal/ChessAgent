import random

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, TensorDataset

from unique_states_experiment.train import get_data
from unique_states_experiment.model1NN import ModelAttention

import matplotlib.pyplot as plt

import os


def test_model(model, dataloader, model_name, mode="Test"):
    """
    Evaluates the model on the provided dataloader by computing the mean squared error (MSE)
    between the output probability distributions and the label distributions.

    Assumes that the model outputs log probabilities (via log_softmax) and the labels are probability distributions.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()  # Uses the default 'mean' reduction

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            outputs, _ = model(batch_inputs)
            probs = torch.exp(outputs)
            loss = criterion(probs, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total_samples += batch_labels.size(0)

    overall_mse = total_loss / total_samples
    print(f"{mode} {model_name} MSE: {overall_mse:.4f}")
    return overall_mse




if __name__ == "__main__":
    seed = 904056181
    generator = torch.Generator().manual_seed(seed)
    inputs, outputs = get_data()
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, output_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=generator)
    model_paths = os.listdir('./models')
    try:
        model_paths.sort(key=lambda x: int(x.split("_")[0]))
    except ValueError:
        pass
    x = []
    mse = []
    for i, model_path in enumerate(model_paths):
        model = ModelAttention.load_model("./models/"+model_path)
        t_mse = test_model(model, test_loader, model_path)
        x.append(i)
        mse.append(t_mse)
    plt.clf()
    plt.plot(x, mse)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("MSE per epoch (863,796 samples)")
    plt.savefig('./data/mse.png', format='png')


