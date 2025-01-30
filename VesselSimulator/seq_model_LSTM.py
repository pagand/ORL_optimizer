import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
import pickle



# Initialize wandb
wandb.init(project="Vessel Simulator" , entity="marslab", name = "LSTM_AR")

# LSTM Model
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.1):
        super(AutoregressiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)  # Apply dropout
        out = self.fc(out)
        return out, hidden

def train_step(model, data_loader, criterion, optimizer, device, current_epoch, total_epochs, max_ar_steps, train=True):
    epoch_loss = 0

    lambda_ar = min(1.0, current_epoch / total_epochs)  # Increase AR importance over epochs
    lambda_nar = 1.0 - lambda_ar  # Decrease reliance on ground-truth

    autoregressive_steps = max(1, int(lambda_ar * max_ar_steps))  # Increase AR steps progressively



    for batch in data_loader:
        inp_batch, out_batch = batch
        inp_batch, out_batch = inp_batch.to(device), out_batch.to(device)

        # Initialize hidden state
        hidden_start = None

        # Process each sequence in the batch
        batch_loss, loss_nar, loss_ar = 0, 0, 0
        count_ar, count_nar = 0, 0
        for t in range(inp_batch.shape[1]):  # True clock time
    
            current_input = inp_batch[:, t, 1:].clone()

            hidden = hidden_start
            
            
            for s in range(t, min(t+autoregressive_steps, inp_batch.shape[1]-1)):  
                done_mask = inp_batch[:, s, 0] == 0
                predictions, hidden = model(current_input.unsqueeze(1), hidden)
                if s == t: # First step: store hidden state
                    hidden_start = hidden
                current_input = current_input.clone()
                current_input[:, -4:] = predictions.squeeze(1)[:,:4]

                target = out_batch[:, s, :].unsqueeze(1)
                if done_mask.sum() > 0:
                    step_loss = criterion(predictions[done_mask], target[done_mask])
                    batch_loss += step_loss

                    if s == t: # NAR loss
                        loss_nar += step_loss
                        count_nar += 1
                    else: # AR loss
                        loss_ar += step_loss
                        count_ar += 1
                        

        # Compute weighted loss
        loss_nar /= count_nar
        loss_ar = loss_ar/count_ar if count_ar else torch.tensor(0).to(device)
        batch_loss /= (count_ar + count_nar)

        final_loss = lambda_nar * loss_nar + lambda_ar * loss_ar  # Weighted loss
        # final_loss = batch_loss   # Unweighted loss

        # Backpropagation and optimizer step
        if train:
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

        epoch_loss += final_loss.item()
        return epoch_loss/ len(data_loader), loss_nar.item(), loss_ar.item()

def main(train_path, save_path, retrainer):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 128
    num_layers = 2
    batch_size = 16
    learning_rate = 0.001
    max_ar_steps = 5 # Steps to predict ahead, 1 for NAR 
    epochs = 40*max_ar_steps # total epochs
    
    # Dataset Preparation (Replace this with your numpy arrays)
    data = pickle.load(open(train_path, 'rb'))
    inp = data['train']['inp']
    out = data['train']['out']

    # crop the seq to avoid zeros
    inp = inp[:, :110, :]
    out = out[:, :110, :]

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
    # update the model params with the best model if retraining
    if retrainer:
        try:
            model.load_state_dict(torch.load(save_path))
            print("Retraining the model from the best version found")
        except:
            print("No model found to retrain")
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training Loop
    best_loss = np.inf
    for epoch in range(epochs):
        model.train()
        epoch_loss, loss_nar, loss_ar = train_step(model, data_loader, criterion, optimizer, device, epoch, epochs, max_ar_steps, train=True)
        # do the test
        model.eval()
        epoch_loss_test, loss_nar_test, loss_ar_test = train_step(model, data_loader_test, criterion, optimizer, device,  epoch, epochs, max_ar_steps, train=False) 
        if epoch_loss_test < best_loss:
            best_loss = epoch_loss_test
            best_model = model
            best_epoch = epoch + 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss / len(data_loader)}, Test Loss: {epoch_loss_test / len(data_loader_test)}")

        # Log metrics to wandb
        wandb.log({"epoch_loss": epoch_loss, "val_loss": epoch_loss_test,
                  "loss_nar": loss_nar, "loss_ar" : loss_ar,
                    "loss_nar_test":loss_nar_test, "loss_ar_test":loss_ar_test,"epoch": epoch + 1})
    print(f"Best model found at epoch {best_epoch}")
    torch.save(best_model.state_dict(), save_path)

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    retrainer = True
    train_path = './data/VesselSimulator/data_train.pkl'
    save_path = './data/VesselSimulator/lstm_model_ar.pth'
    main(train_path, save_path, retrainer)
