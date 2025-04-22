import os
import random
import shutil

# Path to your 'train' folder (contains images and labels folders)
train_folder = r"C:\LPR-5\train"  # <<< CHANGE THIS if needed!

# New folders for splitted data
base_folder = os.path.dirname(train_folder)  # one level up, C:\LPR-5
train_out = os.path.join(base_folder, "train_split")
valid_out = os.path.join(base_folder, "valid_split")

# Create train_split and valid_split directories
for split in ['train_split', 'valid_split']:
    os.makedirs(os.path.join(base_folder, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_folder, split, 'labels'), exist_ok=True)

# Paths
images_path = os.path.join(train_folder, "images")
labels_path = os.path.join(train_folder, "labels")

# List images
images = os.listdir(images_path)
random.shuffle(images)

# 80-20 split
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
valid_images = images[split_idx:]

def move_files(file_list, destination_folder):
    for img_file in file_list:
        img_src = os.path.join(images_path, img_file)
        label_src = os.path.join(labels_path, img_file.replace('.jpg', '.txt'))

        img_dst = os.path.join(destination_folder, 'images', img_file)
        label_dst = os.path.join(destination_folder, 'labels', img_file.replace('.jpg', '.txt'))

        shutil.copy(img_src, img_dst)
        shutil.copy(label_src, label_dst)

move_files(train_images, train_out)
move_files(valid_images, valid_out)

print("Dataset successfully split into train_split/ and valid_split/ folders!")
