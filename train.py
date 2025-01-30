import torch
from generate_data_main import GenerateFlowData
from torch.utils.data import Dataset, DataLoader
from models import FluidNet
import torch.nn as nn
import torch.optim as optim
from animate import animate_flow
from torchsummary import summary


class FluidDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to float tensors
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on ", device)
use_curl = True
"""
Best 
(40, 80),
(0.08, 0.12),
(0.01, 0.02)
sample_freq = 60
"""
gfd = GenerateFlowData(
    (40, 80),
    (0.08, 0.12),
    (0.004, 0.02),
    use_curl=use_curl,
    num_timeframes=100,
    num_samples=256,
)
X_np, Y_np = (
    gfd.get_dataset()
)  # X shape: (32, 200, 60, 120, 4), Y shape: (32, 200, 60, 120, 2)


fluid_dataset = FluidDataset(X_np, Y_np)
# Split the data into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(fluid_dataset))
val_size = len(fluid_dataset) - train_size
# Create DataLoaders for training and validation (no random split)
train_dataset = torch.utils.data.Subset(fluid_dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(
    fluid_dataset, range(train_size, len(fluid_dataset))
)

# Create DataLoaders for training and validation
batch_size = 32  # Adjust based on memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
dataset = {"train": train_loader, "validate": val_loader}

# Initialize Model, Loss, Optimizer
print("Adding model to device")
out_channels = 1 if use_curl else 2
model = FluidNet(
    input_channels=gfd.num_channels, output_channels=out_channels, hidden_channels=12
).to(device)
#  summary(model, input_size=(4, 40, 80))

print("Done")
criterion = nn.MSELoss(reduction="none")  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 50
patience = 4
min_loss = 1000
counter = 0
break_train = False

for epoch in range(epochs):
    for key in dataset:
        dataloader = dataset[key]
        if key == "train":
            model.train()
        else:
            model.eval()
        epoch_loss = 0.0

        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if key == "train":
                optimizer.zero_grad()  # Zero out gradients
            outputs = model(X_batch)  # Forward pass
            loss = (criterion(outputs, Y_batch) * (torch.abs(Y_batch) > 1e-2)).mean()
            if key == "train":
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            epoch_loss += loss.item()

            if batch_idx % 10 == 0 and key == "train":
                print(
                    f"Epoch {epoch+1}/{epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )
        print(f"{key}: Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.6f}")
        if key == "validate":
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                counter = 0
            else:
                counter += 1
        if counter > patience:
            break_train = True
    if break_train:
        print(f"Reached {patience} epochs without validtate improvement.")
        break

model.eval()
num_gifs = 12
for i in range(num_gifs):
    gfd.timeframes = 100
    x_t, y_t = gfd.gen_X_and_Y()
    x_t = (x_t - gfd.mean_X) / gfd.std_X
    y_t = (y_t - gfd.mean_Y) / gfd.std_Y
    x_t[:, -model.output_channels :] = y_t
    barrier = gfd.barrier

    net_y = model.simulate(
        torch.tensor(x_t[0], dtype=torch.float32).to(device=device), 100
    )
    net_y = net_y.cpu().detach().numpy()
    if not use_curl:
        y_t = y_t[:, 0]  # gfd.curl(y_t[:, 0], y_t[:, 1])
        net_y = net_y[:, 0]  # gfd.curl(net_y[:, 0], net_y[:, 1])
    else:
        y_t = y_t[:, 0]

    animate_flow(
        y_t, net_y, barrier.astype(bool), output_gif_path=f"flow_animation{i}.gif"
    )


# Save the trained model
torch.save(model.state_dict(), "fluid_model.pth")
print("Training complete! Model saved as fluid_model.pth")
