import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from utils.NeuralNet import NeuralNet
import pickle

def train_neural_network(train_data, train_target, test_data, test_target, nn_model = NeuralNet(), criterion = torch.nn.MSELoss(), optimizer = torch.optim.Adam, lr = 0.001, batch_size = 64, num_epochs = 100, model_save_name = "x_model.pth", plot=False, n_components=None):
    
    """
    Trains a neural network on the given data and saves the model.
    """
    
    # Applying PCA if n_components is provided
    if n_components is not None:
        pca = PCA(n_components=n_components)
        train_data = pca.fit_transform(train_data)
        test_data = pca.transform(test_data)
        # Convert PCA applied data back to tensors
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
    else:
        pca = None

    # Convert data to tensors if not already
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not isinstance(train_target, torch.Tensor):
        train_target = torch.tensor(train_target, dtype=torch.float32)
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    if not isinstance(test_target, torch.Tensor):
        test_target = torch.tensor(test_target, dtype=torch.float32)

    # Adjust target tensor type based on the loss function
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        train_target = train_target.long()
        test_target = test_target.long()
    elif isinstance(criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)):
        train_target = train_target.float()
        test_target = test_target.float()

    # Create data loaders
    train_dataset = TensorDataset(train_data, train_target)
    test_dataset = TensorDataset(test_data, test_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = nn_model
    optimizer = optimizer(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                # Metrics calculation
                if isinstance(criterion, torch.nn.MSELoss):
                    continue  # RMSE will be calculated after loop
                elif isinstance(criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)):
                    predicted = (output >= 0.5).float()
                elif isinstance(criterion, torch.nn.CrossEntropyLoss):
                    predicted = torch.argmax(output, dim=1)
                
                correct += (predicted == target).sum().item()
                total += target.size(0)
        test_losses.append(test_loss / len(test_loader))

        # Print additional metrics
        if isinstance(criterion, torch.nn.MSELoss):
            rmse = np.sqrt(test_losses[-1])
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test RMSE: {rmse:.4f}")
        else:
            accuracy = correct / total * 100
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {accuracy:.2f}%")

    # Plotting the training and validation losses
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save the model
    torch.save(model.state_dict(), model_save_name)
    print(f"Model saved as {model_save_name}")

    # Save PCA model if applied
    if pca is not None:
        pca_file = model_save_name.replace('.pth', '_pca.pkl')
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
        print(f"PCA model saved as {pca_file}")

    return model, pca
