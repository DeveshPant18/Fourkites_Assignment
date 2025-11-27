import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.models import SimpleCNN
from src.landscape_utils import get_params, get_random_dir, set_weights

def main():
    # 1. Load Data (Use Test set for generalization analysis)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False)
    
    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    criterion = nn.CrossEntropyLoss()
    
    print("Generating Loss Landscape...")
    
    # 3. Setup Directions
    origin_params = get_params(model)
    dir1 = get_random_dir(model) # X-axis direction
    dir2 = get_random_dir(model) # Y-axis direction (for 2D)

    # --- PART A: 1D Linear Interpolation ---
    # We look from -1.0 to 1.0 around the center (0.0 is the trained minimum)
    steps_1d = 21
    x_coords = np.linspace(-1.0, 1.0, steps_1d)
    losses_1d = []

    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(test_loader)) # Use one large batch for speed
        inputs, targets = inputs.to(device), targets.to(device)
        
        for alpha in x_coords:
            set_weights(model, origin_params, dir1, None, alpha, 0)
            output = model(inputs)
            loss = criterion(output, targets).item()
            losses_1d.append(loss)

    # Restore parameters
    set_weights(model, origin_params, dir1, None, 0, 0) 

    # Plot 1D
    plt.figure(figsize=(8, 5))
    plt.plot(x_coords, losses_1d, marker='o', label='Loss Surface')
    plt.title("1D Loss Landscape Visualization")
    plt.xlabel("Step Size (Alpha)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("results_1d.png")
    print("Saved results_1d.png")

    # --- PART B: 2D Contour Plot ---
    # Create a grid of alpha (x) and beta (y)
    steps_2d = 21 
    grid_range = 1.0
    x = np.linspace(-grid_range, grid_range, steps_2d)
    y = np.linspace(-grid_range, grid_range, steps_2d)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    print("Computing 2D Surface (this may take a moment)...")
    with torch.no_grad():
        for i in range(steps_2d):
            for j in range(steps_2d):
                alpha = X[i, j]
                beta = Y[i, j]
                set_weights(model, origin_params, dir1, dir2, alpha, beta)
                
                output = model(inputs)
                Z[i, j] = criterion(output, targets).item()

    # Plot 2D
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cp)
    plt.scatter(0, 0, c='red', marker='x', s=100, label='Optimized Minima')
    plt.title("2D Loss Landscape Contour")
    plt.xlabel("Direction 1")
    plt.ylabel("Direction 2")
    plt.legend()
    plt.savefig("results_2d.png")
    print("Saved results_2d.png")

if __name__ == "__main__":
    main()