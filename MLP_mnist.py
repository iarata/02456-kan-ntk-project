import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import lightning as L

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# MNIST
train_dataset_mnist = MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
train_loader_mnist = DataLoader(train_dataset_mnist)
test_dataset_mnist = MNIST(os.getcwd(), download=True, train=False, transform=transforms.ToTensor())
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=1, shuffle=False)

print("Shape: ", train_dataset_mnist[0][0].size())

class simpleMLP(nn.Module):
    def __init__(self, input_sz, num_classes):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(input_sz, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten while keeping the batch dimension
        x = self.NN(x)
        return x
    
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, targets = batch
        x = self.model(x)
        loss = F.cross_entropy(x, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

### MNIST time ###
# model
lit_model = LitModel(simpleMLP(28*28, 10))

# train model
trainer = L.Trainer(max_epochs=2)
trainer.fit(model=lit_model, train_dataloaders=train_loader_mnist)

# model to device for inference
lit_model.model.to(device)
lit_model.model.eval()

import matplotlib.pyplot as plt

## MNIST ##
plt.figure(figsize=(16, 16))
with torch.no_grad():

    for i, (im, target) in enumerate(test_loader_mnist):
        if i >= 16: 
            break
        im = im.to(device)  
        target = target.to(device) 
        pred = lit_model.model(im)
        pred_label = torch.argmax(pred, dim=1).item() 

        plt.subplot(4, 4, i + 1)
        plt.title(f"Target: {target.item()}, Pred: {pred_label}")
        plt.imshow(im[0][0].cpu().numpy(), cmap="gray")  
        plt.axis('off')

    plt.savefig("./vis/mnist_test.png")
    plt.close()

    test_acc = 0
    for im, target in test_loader_mnist:
        im = im.to(device)  
        target = target.to(device) 
        pred = lit_model.model(im)
        pred_label = torch.argmax(pred, dim=1).item() 
        if (int(target.item()) == int(pred_label)):
            test_acc += 1

    test_acc /= len(test_loader_mnist)
    print(f"Test acc MNIST: {test_acc}")
