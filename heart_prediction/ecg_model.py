import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import pandas as pd
import os

# Custom Dataset for loading ECG images and labels from the provided CSV
class ECGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # Column 0 contains image paths
        image = Image.open(img_path).convert('RGB')  # Ensure images are RGB
        
        label = self.data.iloc[idx, 1]  # Column 1 contains target labels
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations for the dataset (Resizing to 224x224 and normalizing for pretrained models)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load the dataset using the custom ECGDataset class
dataset = ECGDataset(csv_file='ecg_images_dataset.csv', transform=transform)

# Split dataset into training and validation sets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load a pre-trained ResNet model and modify for the ECG task
model = models.resnet50(pretrained=True)

# Freeze early layers (example: freeze up to layer4)
for name, param in model.named_parameters():
    if 'layer4' not in name:  # Fine-tune only layer4 and beyond
        param.requires_grad = False

# Replace the final fully connected layer to match the number of classes (e.g., 4 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Update output to 4 classes

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Early Stopping parameters
patience = 5  # Number of epochs without improvement before stopping
best_loss = float('inf')
early_stop_counter = 0

# Path to save the best model weights
best_model_weights_path = 'best_model_weights.pth'
entire_model_path = 'entire_model.pth'

# Training loop with early stopping and validation
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Training phase
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # Calculate average training loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            
            # Forward pass on validation data
            val_outputs = model(val_inputs)
            v_loss = criterion(val_outputs, val_labels)
            
            val_loss += v_loss.item()
            
            # Calculate validation accuracy
            _, predicted = torch.max(val_outputs, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()
    
    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    # Save best model weights based on validation loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stop_counter = 0  # Reset counter
        torch.save(model.state_dict(), best_model_weights_path)  # Save best weights
        print(f"Best model weights saved at epoch {epoch+1}")
    else:
        early_stop_counter += 1
        print(f"Early stop counter: {early_stop_counter}")
        
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

# Save the entire model after training is complete
torch.save(model, entire_model_path)
print("Entire model saved.")
print("Training completed.")
