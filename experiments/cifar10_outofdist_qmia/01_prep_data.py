import os
import shutil

def combine_datasets(train_dir, val_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    def copy_images(src_dir):
        for root, _, files in os.walk(src_dir):
            # Compute the relative path from the source directory
            rel_path = os.path.relpath(root, src_dir)
            # Determine the destination folder path
            dest_folder = os.path.join(output_dir, rel_path) if rel_path != '.' else output_dir
            os.makedirs(dest_folder, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, file)
                shutil.copy2(src_file, dest_file)
    
    # Copy images from train and validation directories
    copy_images(train_dir)
    copy_images(val_dir)

if __name__ == '__main__':
    train_dir = os.path.join('data', 'processed', 'cifar10', 'train')
    val_dir = os.path.join('data', 'processed', 'cifar10', 'val')
    output_dir = os.path.join('data', 'temp', 'cifar10_trainval_combo')
    
    combine_datasets(train_dir, val_dir, output_dir)
    print(f"Combined dataset created at {output_dir}")