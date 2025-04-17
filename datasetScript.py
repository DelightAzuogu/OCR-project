import os
import csv

# Set the path to the main folder containing all the smaller folders
big_folder_path = "dataset"  # Replace with your actual path if needed
output_csv_path = "train.csv"

# Define valid image extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image', 'label'])  # Write header

    # Walk through each folder inside the big folder
    for folder_name in os.listdir(big_folder_path):
        small_folder_path = os.path.join(big_folder_path, folder_name)
        print(f"Processing folder: {small_folder_path}")

        if os.path.isdir(small_folder_path):
            label = folder_name[0]  # Take only the first character
            for file_name in os.listdir(small_folder_path):
                file_path = os.path.join(small_folder_path, file_name)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file_name)
                    if ext.lower() in image_extensions:
                        # Use the exact path as it appears in the filesystem
                        relative_path = os.path.join(big_folder_path, folder_name, file_name)
                        writer.writerow([relative_path, label])

print(f"CSV file created at: {output_csv_path}")