import os
import pandas as pd
import random
import shutil
from pathlib import Path
import argparse

def split_train_val(train_folder, train_csv, val_size=0.2, seed=42):
    """
    Split the training dataset into train and validation sets.
    
    Parameters:
    -----------
    train_folder : str
        Path to the folder containing training images
    train_csv : str
        Path to the CSV file with annotations
    val_size : float
        Proportion of data to use for validation (default: 0.2)
    seed : int
        Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create validation folder if it doesn't exist
    val_folder = os.path.join(os.path.dirname(train_folder), 'val')
    os.makedirs(val_folder, exist_ok=True)
    
    # Read the training CSV
    df = pd.read_csv(train_csv)
    
    # Get unique image IDs
    unique_images = df['image_id'].unique()
    print(f"Total unique images: {len(unique_images)}")
    
    # Calculate number of images for validation
    val_count = int(len(unique_images) * val_size)
    print(f"Images to move to validation: {val_count}")
    
    # Randomly select images for validation
    val_images = random.sample(list(unique_images), val_count)
    print(f"Selected {len(val_images)} images for validation")
    
    # Split DataFrame
    val_df = df[df['image_id'].isin(val_images)].copy()
    train_df = df[~df['image_id'].isin(val_images)].copy()
    
    print(f"Training entries: {len(train_df)}")
    print(f"Validation entries: {len(val_df)}")
    
    # Move images from train to val folder
    moved_images = 0
    for img_id in val_images:
        # Some datasets might have extensions like .jpg, .png, etc.
        # Try common extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        found = False
        
        for ext in extensions:
            src_path = os.path.join(train_folder, f"{img_id}{ext}")
            if os.path.exists(src_path):
                dst_path = os.path.join(val_folder, f"{img_id}{ext}")
                shutil.copy2(src_path, dst_path)
                # Option to remove from train folder
                # os.remove(src_path)
                moved_images += 1
                found = True
                break
                
        # If no extension worked, try finding the file by listing directory
        if not found:
            for file in os.listdir(train_folder):
                if file.startswith(f"{img_id}.") or file == img_id:
                    src_path = os.path.join(train_folder, file)
                    dst_path = os.path.join(val_folder, file)
                    shutil.copy2(src_path, dst_path)
                    # Option to remove from train folder
                    # os.remove(src_path)
                    moved_images += 1
                    break
    
    print(f"Moved {moved_images} images to validation folder")
    
    # Save the updated CSVs
    train_csv_name = os.path.basename(train_csv)
    base_name, ext = os.path.splitext(train_csv_name)
    
    train_output = os.path.join(os.path.dirname(train_csv), f"{base_name}_updated{ext}")
    val_output = os.path.join(os.path.dirname(train_csv), f"val{ext}")
    
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    
    print(f"Updated training CSV saved to: {train_output}")
    print(f"Validation CSV saved to: {val_output}")
    
    return train_output, val_output

if __name__ == "__main__":
    # Hard-coded parameters
    train_folder = "train"  # Path to your train folder
    train_csv = "train.csv"  # Path to your train.csv file
    val_size = 0.2  # 20% of data for validation
    seed = 42  # Random seed for reproducibility
    
    # Call the function with the specified parameters
    split_train_val(train_folder, train_csv, val_size, seed)