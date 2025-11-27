import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from src.models import SimpleCNN

def train():
    # 1. Setup Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 3. Train Loop
    print(f"Training on {device}...")
    model.train()
    for epoch in range(5): # Short training for demo
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

    # 4. Save Model
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved to trained_model.pth")

if __name__ == "__main__":
    train()