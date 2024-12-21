import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from typing import Iterable

import torch
import torch.nn as nn

from matplotlib.animation import FuncAnimation
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def pkernel_loss(y_pred: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
    """
    Computes the loss function with one-hot y_preds and normalizes by number of classes,
    pkernel computes the loss over all possible targets, as opposed to over e.g. the actual
    targets, this is helpful for forming the ntk matrix.

    Args:
        y_pred: the predictions of the model i.e. model(x), [N_samples, C] for C classes
        loss: the loss function we use
    """
    # Variable to hold the loss
    total_loss = 0.
    # The batch size
    B = y_pred.shape[0]
    # The number of classes in the dataset
    n_classes = y_pred.shape[-1]
    # For each class, compute the loss and add it to total loss
    for i in range(n_classes):
        """
        sps i = 2, n = 5 and C = 3, then the following line can be broken down as follows

        torch.zeros_like(y_pred) --->
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
        torch.tensor([i], dtype=torch.long, device=y_pred.device) --->
        [2]
        .repeat(B)
        [2, 2, 2]
        .unsqueeze(1)
        [[2], [2], [2]]
        scatter(1, ..., 1)
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]]
        """
        y_target = torch.zeros_like(y_pred).scatter(1, torch.tensor([i], dtype=torch.long, device=y_pred.device).repeat(B).unsqueeze(1), 1)

        # Some loss function that works on one-hot encoded targets
        total_loss += loss(y_pred.float(), y_target)

    # Divide loss by sqrt(n_classes), this is just for stabization and bcs it appears in theory
    # In theory, dividing by # classes should ofc not do anything to the optimum or the gradient direction
    total_loss = total_loss / n_classes**0.5
    return total_loss

def get_gradient(model: torch.nn.Module, x: torch.Tensor, loss_fn: callable, opt: torch.optim.Optimizer, flatten: bool = False, use_label: bool = False, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Get the gradient of each element of the batch in x of the model with respect to the loss function
    
    Args:
        model: the model to use
        x: the input data [N_samples, ...]
        y: the target data [N_samples, ...]
        loss_fn: the loss function to use
        opt: the optimizer to use
        positive: whether to perturb the loss function positively or negatively
        kernel: What type of kernel to use, can either be pKernel, direct or average
        
    Returns:
        grads: the gradients of the model with respect to the loss function
    """
    # Number of samples / Batch size
    B = len(x) 
    # GPU or CPU
    device = next(model.parameters()).device
    x = x.to(device)
    if y is not None:
        y = y.to(device).float()
    # Zero any existing gradients, then compute predictions on the input data
    opt.zero_grad()
    y_pred = model(x)
    if use_label:
        y_target = y
    else:
        y_target = y_pred.detach().argmax(1)
        y_target = torch.zeros_like(y_pred).scatter(1, y_target.unsqueeze(1), 1)
    
    # Compute the kernel_loss based on the loss function used
    loss = pkernel_loss(y_pred, loss_fn)
    # Trick to get gradient with respect to each sample in parallel
    grads = torch.autograd.grad(loss, model.parameters(), is_grads_batched=True, grad_outputs=torch.eye(B).to(device))
    if flatten:
        grads = torch.cat([grad.view((B, -1)) for grad in grads], -1)
    return grads

class own_linear_layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(own_linear_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.beta = 0.1
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize the weights to normal distribution
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=1.0)
    
    def forward(self, x):
        return x @ self.weight.t()/(self.weight.shape[-1]**0.5) + self.beta*self.bias

class SingleLayerMLP(nn.Module):
    """ A simple single hidden-layer perceptron for MNIST classification """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int = 1):
        super(SingleLayerMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Define layers
        self.layers = nn.ModuleList()
        self.layers.append(own_linear_layer(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(own_linear_layer(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(own_linear_layer(hidden_size, output_size))
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, own_linear_layer):
                layer.reset_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input
        x = x.view(-1, self.input_size)
        
        # Forward pass
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_weights(self):
        return [layer.weight for layer in self.layers if isinstance(layer, own_linear_layer)]
    
    def get_biases(self):
        return [layer.bias for layer in self.layers if isinstance(layer, own_linear_layer)]
    
    def set_weights(self, weights, biases, initial_gain):
        i = 0
        for layer in self.layers:
            if isinstance(layer, own_linear_layer):
                weight = weights[i]
                bias = biases[i]
                in_features = layer.in_features
                out_features = layer.out_features
                k_factor = initial_gain
                layer.weight.data = weight.data[:out_features, :in_features]*k_factor
                layer.bias.data = bias.data[:out_features]*k_factor
                i += 1
    

class GaussianFit(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: torch.device, kernel_method: str = "direct", noise_var: float = 0.0):
        super(GaussianFit, self).__init__()
        self.device = device
        self.model = model
        self.kernel_method = kernel_method
        self.noise_var = noise_var
        self.covariance_matrix = None
        self.optimizer = None
        
    def fit(self, data: Iterable[torch.Tensor], optimizer: torch.optim.Optimizer, loss_batched: torch.nn.Module):
        self.optimizer = optimizer
        self.loss = loss_batched
        xs, ys, y_hats = [], [], []
        with torch.no_grad():
            for x, y in data:
                xs.append(x)
                ys.append(y)
                x.to(self.device)
                y_hats.append(self.model(x))
        xs = torch.cat(xs, 0).to(self.device)
        y = torch.cat(ys, 0).to(self.device)
        y_hat = torch.cat(y_hats, 0).to(self.device)
        self.label_diff = y - y_hat
        self.grads = get_gradient(self.model, xs, loss_batched, self.optimizer, True, True, y=y).to(self.device) # Added .to(self.device)
        self.update_w()
        
    def update_noise(self, noise_var: float):
        self.noise_var = noise_var
        self.update_w()
        
    def update_w(self):
        self.covarinace_kernel = self.grads@self.grads.T
        self.covariance_matrix = self.covarinace_kernel.clone()
        self.covariance_matrix[range(self.covariance_matrix.shape[0]), range(self.covariance_matrix.shape[0])] += self.noise_var
        self.W = torch.linalg.solve(self.covariance_matrix.cpu(), self.label_diff.cpu()).to(self.device)
        
    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        """ Function transforming input x into the gradient kernel space """
        x_grad = get_gradient(self.model, x, self.loss, self.optimizer, True, True)
        return x_grad @ self.grads.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_grad = get_gradient(self.model, x, self.loss, self.optimizer, True, True)
        K_xX = x_grad@self.grads.T
        with torch.no_grad():
            y_hat = self.model(x)
        return y_hat + K_xX @ self.W
 
def MSELoss_batch(y_hat, y):
    return 0.5*(y_hat-y).pow(2).sum(-1)


def decision_boundary(x: torch.Tensor, treshold: float, flip_chance: float = 0) -> torch.Tensor:
    y = ((x[:, 0] > treshold) | (x[:, 1] > treshold)).float()
    should_flip = torch.rand(y.size()) < flip_chance
    y[should_flip] = 1 - y[should_flip]
    return y

def sample_data(n: int, treshold: float = 0.5**0.5, seed: Optional[int] = None, flip_chance: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.rand(n, 2)
    y = decision_boundary(x, treshold, flip_chance)
    return x, y


# def plot_decision_boundary(model: torch.nn.Module, treshold: float, fig, ax, title = None) -> None:
#     model.eval()
#     y_hat = torch.stack([model.forward(x_test_batch) for x_test_batch in torch.split(x_test, 100)]).view(-1)
#     y_acc = (y_hat >= 0.5)== ((x_test[:, 0] > treshold) | (x_test[:, 1] > treshold))
#     ax.clear()
    
#     y_hat_reshape = y_hat.view(int(N_testing**0.5), int(N_testing**0.5)).detach().numpy().reshape(int(N_testing**0.5), int(N_testing**0.5))
#     # Plot decision boundary
#     # Color blue if y_hat > 0.5, red if y_hat < 0.5 by making a colormap of two colors
#     colors = ["tab:red", "tab:blue"]
#     custom_cmap = plt.cm.colors.ListedColormap(colors)
#     cb = ax.contourf(x0, x1, y_hat_reshape >0.5, alpha=0.5, levels=torch.linspace(-5.5, 5.5, 3), cmap=custom_cmap)
     
#     # Plot training data
#     ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], c='r', label='Class 0', s=4, marker='x')
#     ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], c='b', label='Class 1', s=4, marker='x')
#     # Add title
#     if title is not None:
#         ax.set_title(title)
#     else:
#         ax.set_title('Accuracy: {:.2f}'.format(y_acc.float().mean().item()))
#     ax.set_xlabel('$x_0$')
#     ax.set_ylabel('$x_1$')
    
#     # Add striped line to mark the decision boundary (x0 < treshold and x1 < treshold)
#     ax.plot([0, treshold], [treshold, treshold], 'k--')
#     ax.axvline(x=treshold, color='k', linestyle='--', ymax=treshold, label='Correct Decision boundary')
#     ax.legend(loc='upper center')
#     model.train()
    
# def plot_NTK_decision_boundary(n_hidden: int = 64, noise_var: float = 0.0, ax: Optional[plt.Axes] = None, fig: Optional[plt.Figure] = None, title = None):
#     n_hidden = int(n_hidden)
#     model = train_model  # model_arch(2, 1, n_hidden)

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
#     kernel_model = GaussianFit(model, "cpu", noise_var=noise_var)
#     kernel_model.fit(gradient_loader, optimizer, MSELoss_batch)
#     plot_decision_boundary(kernel_model, treshold, fig, ax, title)
#     plt.show()


# ##### START OF DEMO #####
# # Setup data
# treshold = 0.5**0.5
# flip_chance = 0.0
# N_training = 100
# N_testing = 10**4
# x_train, y_train = sample_data(N_training, seed=0, flip_chance=flip_chance, treshold=treshold)
# # Construct grid of sqrt(N_testing) x sqrt(N_testing) points
# x = torch.linspace(0, 1, int(N_testing**0.5))
# x0, x1 = torch.meshgrid(x, x, indexing='ij')
# x_test = torch.stack([x0.flatten(), x1.flatten()], 1)
# y_test = decision_boundary(x_test, treshold)

# # Construct NN_model
# n_hidden = 64*128
# model_arch = SingleLayerMLP
# criterion = lambda x, y: ((x-y)**2).mean()  # We need L2 loss for NTK
# gradient_loader = torch.utils.data.DataLoader([(x_train[i], y_train[i, None]) for i in range(N_training)], batch_size=64, shuffle=False)
# train_model = model_arch(2, 1, n_hidden)
# optimizer = torch.optim.SGD(train_model.parameters(), lr=1., momentum=0.9)

# # Train network to compare decision boundaries 
# train_model.train()
# for i in tqdm(range(10000)):
#     for x, y in gradient_loader:
#         optimizer.zero_grad()
#         y_hat = train_model(x)
#         loss = ((y_hat - y.float())**2).mean()
#         loss.backward()
#         optimizer.step()
        
    
# # Plot decision boundary
# fig, ax = plt.subplots(2, 1)
# fig.set_size_inches(5, 10)
# plot_decision_boundary(train_model, treshold, fig, ax[0], 'Trained NN')

# plot_NTK_decision_boundary(n_hidden, 0.0, ax[1], fig, 'Corresponding Gaussian Process')
# fig.tight_layout()
# fig.patch.set_alpha(0)  # Make background transparent

# # Save figure
# fig.savefig('decision_boundary_comparison.svg')
# plt.show()