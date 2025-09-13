import json
import os

def create_image_list(coco_annotation_file, output_txt_file):
    # Read the COCO annotation file
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Extract image filenames and remove file extensions
    image_names = [os.path.splitext(image_info['file_name'])[0] for image_info in coco_data['images']]
    
    # Ensure the output directory exists; create it if it does not
    output_dir = os.path.dirname(output_txt_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the image names to the specified TXT file
    with open(output_txt_file, 'w') as f:
        for image_name in image_names:
            f.write(image_name + '\n')

def create_train_and_val_txt(train_json, val_json, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the training set TXT file
    train_txt_path = os.path.join(output_dir, 'train.txt')
    create_image_list(train_json, train_txt_path)
    
    # Create the validation set TXT file
    val_txt_path = os.path.join(output_dir, 'val.txt')
    create_image_list(val_json, val_txt_path)

# Example usage
train_file = '/your_project_dir/labels_train.json'  # Path to the COCO-format file for the training set
val_file = '/your_project_dir/labels_val.json'      # Path to the COCO-format file for the validation set
output_dir = '/your_dataset/Main'  # Output directory