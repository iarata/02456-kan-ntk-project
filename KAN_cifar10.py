import os
import torch
from kan import *
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# CIFAR10
train_dataset_cifar10 = CIFAR10(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
test_dataset_cifar10 = CIFAR10(os.getcwd(), download=True, train=False, transform=transforms.ToTensor())

# Prepare data
train_images = torch.stack([train_dataset_cifar10[i][0] for i in range(20_000)])
train_labels = torch.tensor([train_dataset_cifar10[i][1] for i in range(20_000)])
test_images = torch.stack([test_dataset_cifar10[i][0] for i in range(3_000)])
test_labels = torch.tensor([test_dataset_cifar10[i][1] for i in range(3_000)])

# Flatten images
train_images = train_images.view(train_images.size(0), -1).to(device)
test_images = test_images.view(test_images.size(0), -1).to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)

dataset = {
    "train_input": train_images,
    "test_input": test_images,
    "train_label": train_labels,
    "test_label": test_labels,
}

model = KAN(width=[32*32*3, 64, 10], grid=5, k=3, seed=0, device=device)

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

# Train the model
results = model.fit(
    dataset=dataset,
    opt="LBFGS",
    steps=5,
    metrics=(train_acc, test_acc),
    loss_fn=torch.nn.CrossEntropyLoss(),
    lr=1e-4
)

print(results)
print(f"Train Accuracy: {train_acc().item() * 100:.2f}%")
print(f"Test Accuracy: {test_acc().item() * 100:.2f}%")
