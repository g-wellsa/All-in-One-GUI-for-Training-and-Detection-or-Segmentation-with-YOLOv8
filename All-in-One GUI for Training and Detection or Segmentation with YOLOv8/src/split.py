import os
import shutil
from random import sample

# Function to ensure directories exist
def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to safely copy files
def safe_copy(src, dst):
    try:
        shutil.copy(src, dst)
    except FileNotFoundError as e:
        print(f"File not found: {src}")
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")

# Function to move files
def move_files(files, src_dir, dst_image_dir, dst_label_dir):
    for file in files:
        file_xml = file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(src_dir, file)) and os.path.exists(os.path.join(src_dir, file_xml)):
            safe_copy(os.path.join(src_dir, file), os.path.join(dst_image_dir, file))
            safe_copy(os.path.join(src_dir, file_xml), os.path.join(dst_label_dir, file_xml))
        else:
            print(f"Files {file} and/or {file_xml} do not exist in {src_dir}")

# Setup directories and ratio
crs_path = 'ng'
train_image_path = 'train/images'
train_label_path = 'train/labels'
val_image_path = 'valid/images'
val_label_path = 'valid/labels'
train_ratio = 0.8

# Create necessary directories
create_directories([train_image_path, train_label_path, val_image_path, val_label_path])

# Get list of files
all_files = os.listdir(crs_path)
imgs = [f for f in all_files if f.endswith('.jpg')]
xmls = [f.replace('.jpg', '.txt') for f in imgs]

# Calculate the number of training and validation files
total_imgs = len(imgs)
train_count = int(total_imgs * train_ratio)
val_count = total_imgs - train_count

# Split the files
train_files = sample(imgs, train_count)
val_files = [file for file in imgs if file not in train_files]

# Move files to train and validation directories
move_files(train_files, crs_path, train_image_path, train_label_path)
move_files(val_files, crs_path, val_image_path, val_label_path)

print(f"Training images are: {train_count}")
print(f"Validation images are: {val_count}")
