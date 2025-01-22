import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
import pickle

# Initialize wandb
# wandb.init(project="autoregressive_lstm")

# LSTM Model
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(AutoregressiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

def train_step(model, data_loader, criterion, optimizer, device, autoregressive_steps, train=True):
    epoch_loss = 0
    for batch in data_loader:
        inp_batch, out_batch = batch
        inp_batch, out_batch = inp_batch.to(device), out_batch.to(device)

        # Initialize hidden state
        hidden = None

        # compute non-autoregressive for the whole sequence
        # out_nar, _ = model(inp_batch, hidden)

        # Process each sequence in the batch
        batch_loss = 0
        count = 0
        for t in range(inp_batch.shape[1]):  # True clock time
    
            current_input = inp_batch[:, t, 1:].clone()
            
            
            for s in range(t, min(t+autoregressive_steps, inp_batch.shape[1]-1)):  
                done_mask = inp_batch[:, s, 0] == 0
                predictions, hidden = model(current_input.unsqueeze(1), hidden)
                current_input = current_input.clone()
                current_input[:, -4:] = predictions.squeeze(1)[:,:4]

                target = out_batch[:, s, :].unsqueeze(1)
                if done_mask.sum() > 0:
                    count += 1
                    step_loss = criterion(predictions[done_mask], target[done_mask])
                    batch_loss += step_loss

        # Backpropagation and optimizer step
        if train:
            optimizer.zero_grad()
            (batch_loss /count ).backward()
            optimizer.step()

        epoch_loss += batch_loss.item()
        return epoch_loss

def main(train_path):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 128
    num_layers = 2
    batch_size = 16
    learning_rate = 0.001
    epochs = 50
    autoregressive_steps = 5 # Steps to predict ahead
    # Dataset Preparation (Replace this with your numpy arrays)
    data = pickle.load(open(train_path, 'rb'))
    inp = data['train']['inp']
    out = data['train']['out']

    in_test = data['test']['inp']
    out_test = data['test']['out']


    inp_tensor = torch.tensor(inp, dtype=torch.float32)
    out_tensor = torch.tensor(out, dtype=torch.float32)
    dataset = TensorDataset(inp_tensor, out_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # same for test
    inp_tensor_test = torch.tensor(in_test, dtype=torch.float32)
    out_tensor_test = torch.tensor(out_test, dtype=torch.float32)
    dataset_test = TensorDataset(inp_tensor_test, out_tensor_test)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    # Model, Loss, Optimizer
    input_size = inp.shape[2] - 1  # Exclude done state 
    output_size = out.shape[2]
    model = AutoregressiveLSTM(input_size, output_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training Loop
    best_loss = np.inf
    for epoch in range(epochs):
        model.train()
        epoch_loss = train_step(model, data_loader, criterion, optimizer, device, autoregressive_steps, train=True)
        # do the test
        model.eval()
        epoch_loss_test = train_step(model, data_loader_test, criterion, optimizer, device, autoregressive_steps, train=False) 
        if epoch_loss_test < best_loss:
            best_loss = epoch_loss_test
            best_model = model
            best_epoch = epoch + 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss / len(data_loader)}, Test Loss: {epoch_loss_test / len(data_loader_test)}")

        # Log metrics to wandb
        # wandb.log({"epoch_loss": epoch_loss / len(data_loader), "epoch": epoch + 1})
    print(f"Best model found at epoch {best_epoch}")
    # torch.save(best_model.state_dict(), "./data/lstm_model.pth")

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    train_path = './data/VesselSimulator/data_train.pkl'
    main(train_path)
