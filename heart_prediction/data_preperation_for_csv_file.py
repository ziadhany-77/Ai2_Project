import os
import csv

# Paths to directories containing the images
data_dirs = {
    "heart_images_dataset/Normal_Person_ECG_Images(284x12=3408)": 0,
    "heart_images_dataset/ECG_Images_of_Patient_that_have_abnormal_heartbeat(233x12=2796)": 1,
    "heart_images_dataset/ECG_Images_of_Myocardial_Infarction_Patients(240x12=2880)": 2,
    "heart_images_dataset/ECG_Images_of_Patient_that_have_History_of_MI(172x12=2064)": 3
}

# Output CSV file
csv_file = 'ecg_images_dataset.csv'

# Open CSV file to write
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['image_path', 'target'])
    
    # Loop over each directory and its corresponding target
    for dir_name, target in data_dirs.items():
        # List all files in the directory
        for img_file in os.listdir(dir_name):
            # Full path to the image file
            img_path = os.path.join(dir_name, img_file)
            # Check if it's an image file (e.g., ends with .jpg, .png)
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                # Write the image path and its target label to the CSV file
                writer.writerow([img_path, target])

print(f"CSV file '{csv_file}' created successfully.")
