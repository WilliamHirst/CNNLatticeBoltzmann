import torch
from generate_data_main import GenerateFlowData
from torch.utils.data import Dataset, DataLoader, random_split
from models import FluidNet
import torch.nn as nn
import torch.optim as optim
from animate import animate_flow
import random


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
gfd = GenerateFlowData(
    (40, 100),
    (0.12, 0.12),
    (0.01, 0.02),
    use_curl=use_curl,
    num_timeframes=32 * 3,
    num_samples=20,
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("---- DataLoaders Ready ----")

# Initialize Model, Loss, Optimizer
print("Adding model to device")
out_channels = 1 if use_curl else 2
model = FluidNet(input_channels=gfd.num_channels, output_channels=out_channels).to(
    device
)
print("Done")
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Function to compute the loss (used for both training and validation)
def compute_loss(
    model,
    data_loader,
    criterion,
    device,
    is_training=True,
    accumulation_steps=1,
):
    total_loss = 0
    if is_training:
        optimizer.zero_grad()  # Zero out gradients

    # Loop over batches
    for batch_idx, (X_batch, Y_batch) in enumerate(data_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Simulate the forward process
        num_forward = 2
        X_updated = X_batch
        batch_loss = 0
        for j in range(num_forward):  # Change the inner loop index to 'j'
            outputs = model(X_updated)  # Forward pass
            X_updated = X_batch.clone()
            X_updated[:, -model.output_channels :] = outputs.unsqueeze(1)
            if j:
                batch_loss += criterion(
                    outputs[:-j], Y_batch[j:]
                )  # Adjust for loss computation
            else:
                batch_loss += criterion(outputs, Y_batch)

        total_loss += batch_loss

        if is_training:
            batch_loss.backward()  # Backpropagation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()

    # Clean up
    del X_batch, Y_batch, X_updated  # Delete batches after use
    torch.cuda.empty_cache()  # Release memory

    return total_loss / len(data_loader)  # Average loss


# Early Stopping Setup
epochs = 3
best_val_loss = float("inf")
patience = 2  # Number of epochs to wait for improvement
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()
    train_loss = compute_loss(model, train_loader, criterion, device, is_training=True)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")

    # Validate the model
    model.eval()
    with torch.no_grad():
        val_loss = compute_loss(model, val_loader, criterion, device, is_training=False)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0  # Reset the counter
        # Save the model with the best validation loss
        torch.save(model.state_dict(), "best_fluid_model.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered. Validation loss did not improve for {patience} epochs."
            )
            break

# Load the best model (in case training stopped early)
model.load_state_dict(torch.load("best_fluid_model.pth"))
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
        y_t,
        net_y,
        barrier.astype(bool),
        output_gif_path=f"BAAAAAD_flow_animation{i}.gif",
    )

# Save the final model after training
torch.save(model.state_dict(), "fluid_model.pth")
print("Training complete! Model saved as fluid_model.pth")
