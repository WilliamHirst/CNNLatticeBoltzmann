import torch
from generate_data import GenerateFlowData
from torch.utils.data import Dataset, DataLoader
from models import FluidNet
import torch.nn as nn
import torch.optim as optim
from animate import animate_flow
from torchsummary import summary


# Custom Dataset for Fluid Dynamics Data
class FluidDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to float tensors
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Set device for computation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on", device)

# Generate flow data parameters
use_curl = True  # Determines whether to use curl-based representation

gfd = GenerateFlowData(
    domain_size=(40, 80),  # Grid dimensions
    velocity_range=(0.08, 0.12),  # Velocity magnitude range
    viscosity_range=(0.004, 0.02),  # Viscosity range
    use_curl=use_curl,
    num_timeframes=100,  # Number of timeframes per simulation
    num_samples=256,  # Total number of samples
)

# Generate dataset
X_np, Y_np = gfd.get_dataset()

# Wrap data in a PyTorch Dataset
fluid_dataset = FluidDataset(X_np, Y_np)

# Split data into training (80%) and validation (20%)
train_size = int(0.8 * len(fluid_dataset))
val_size = len(fluid_dataset) - train_size
train_dataset = torch.utils.data.Subset(fluid_dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(
    fluid_dataset, range(train_size, len(fluid_dataset))
)

# DataLoaders for training and validation
batch_size = 32  # Adjust based on available memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
dataset = {"train": train_loader, "validate": val_loader}

# Initialize Model, Loss Function, and Optimizer
print("Adding model to device")
out_channels = 1 if use_curl else 2  # Define output channels based on curl setting
model = FluidNet(
    input_channels=gfd.num_channels, output_channels=out_channels, hidden_channels=12
).to(device)

criterion = nn.MSELoss(reduction="none")  # Mean Squared Error loss
optimizer = optim.Adam(
    model.parameters(), lr=0.001
)  # Adam optimizer with learning rate 0.001

# Training Loop Configuration
epochs = 50
patience = 4  # Early stopping patience
min_loss = float("inf")  # Initialize with a high value
counter = 0  # Early stopping counter
break_train = False  # Flag to break training if no improvement

# Training Loop
for epoch in range(epochs):
    for phase in ["train", "validate"]:
        dataloader = dataset[phase]
        model.train() if phase == "train" else model.eval()
        epoch_loss = 0.0

        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if phase == "train":
                optimizer.zero_grad()

            outputs = model(X_batch)  # Forward pass
            loss = (criterion(outputs, Y_batch) * (torch.abs(Y_batch) > 1e-2)).mean()

            if phase == "train":
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0 and phase == "train":
                print(
                    f"Epoch {epoch+1}/{epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )

        print(f"{phase}: Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.6f}")

        # Early stopping based on validation loss
        if phase == "validate":
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                counter = 0
            else:
                counter += 1

        if counter > patience:
            break_train = True

    if break_train:
        print(
            f"Early stopping: No improvement in validation loss for {patience} epochs."
        )
        break

# Model Evaluation and Visualization
model.eval()
num_gifs = 12  # Number of simulations to generate animations
for i in range(num_gifs):
    gfd.timeframes = 100
    x_t, y_t = gfd.gen_X_and_Y()
    x_t = (x_t - gfd.mean_X) / gfd.std_X
    y_t = (y_t - gfd.mean_Y) / gfd.std_Y
    x_t[:, -model.output_channels :] = y_t
    barrier = gfd.barrier

    # Simulate the flow using trained model
    net_y = model.simulate(
        torch.tensor(x_t[0], dtype=torch.float32).to(device=device), 100
    )
    net_y = net_y.cpu().detach().numpy()

    if not use_curl:
        y_t = y_t[:, 0]  # Extract x-component if not using curl
        net_y = net_y[:, 0]
    else:
        y_t = y_t[:, 0]

    # Generate animation
    animate_flow(
        y_t, net_y, barrier.astype(bool), output_gif_path=f"flow_animation{i}.gif"
    )

# Save trained model
torch.save(model.state_dict(), "fluid_model.pth")
print("Training complete! Model saved as fluid_model.pth")
