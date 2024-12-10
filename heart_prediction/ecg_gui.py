import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os

# Set up customtkinter for dark/light theme
ctk.set_appearance_mode("Dark")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue", "green", "dark-blue"

# Define the image transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Function to load the trained model
def load_model(weights_path):
    model = torch.load('entire_model.pth')  # Load the entire model (replace with correct path)
    model.load_state_dict(torch.load(weights_path))  # Load the best weights
    model.eval()  # Set model to evaluation mode
    return model

# Function to classify an image using the loaded model
def classify_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Example classes (you can replace these with actual classes)
    classes = ['Normal Heart Beat', 'Abnormal Heart Beat', 'Having A Heart attack', 'Had A Heart Attack']
    
    return classes[predicted.item()]

# Function to handle image upload and prediction
def open_file_and_classify():
    file_path = filedialog.askopenfilename(
        title="Select an ECG Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    if file_path:
        try:
            display_image(file_path)  # Display selected image
            prediction = classify_image(model, file_path)
            prediction_label.configure(text=f"Prediction: {prediction}")  # Update prediction label
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("File Error", "No file selected!")

# Function to display the selected image in the GUI
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Resize image to fit into the label
    img_tk = ImageTk.PhotoImage(img)

    image_label.configure(image=img_tk)  # Use configure() instead of config()
    image_label.image = img_tk  # Keep a reference to avoid garbage collection

# GUI setup
app = ctk.CTk()
app.geometry("800x600")  # Increased window size
app.title("ECG Image Classifier")

# Load the model and weights
weights_path = 'best_model_weights.pth'  # Path to the best model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(weights_path)
model.to(device)

# Create and arrange GUI components

# Title Label
title_label = ctk.CTkLabel(app, text="ECG Image Classifier", font=("Helvetica", 24, "bold"))
title_label.pack(pady=30)

# Frame for Image and Buttons
frame = ctk.CTkFrame(app)
frame.pack(pady=20)

# Image Display Label (starts empty)
image_label = ctk.CTkLabel(frame, text="Upload an ECG image", width=300, height=300)
image_label.grid(row=0, column=0, padx=20, pady=20)

# Upload and Classify Button
upload_button = ctk.CTkButton(frame, text="Upload and Classify Image", font=("Arial", 16), width=200, height=50, command=open_file_and_classify)
upload_button.grid(row=0, column=1, padx=20, pady=20)

# Prediction Display Label (starts empty)
prediction_label = ctk.CTkLabel(app, text="Prediction: ", font=("Arial", 16))
prediction_label.pack(pady=20)

# Footer with Information
footer_label = ctk.CTkLabel(app, text="Upload an ECG image to classify using the trained model", font=("Arial", 14))
footer_label.pack(pady=20)

# Run the app
app.mainloop()
