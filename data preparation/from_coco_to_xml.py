import json
import os
import xml.etree.ElementTree as ET

def create_voc_xml(image_info, annotations, output_dir):
    # Create the root element
    annotation = ET.Element("annotation")

    # Add folder and filename
    folder = ET.SubElement(annotation, "folder")
    folder.text = "VOC2007"
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_info['file_name']
    
    # Add source information
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "The VOC2007 Database"
    annotation_type = ET.SubElement(source, "annotation")
    annotation_type.text = "PASCAL VOC2007"
    
    # Add owner information
    owner = ET.SubElement(annotation, "owner")
    flickr_id = ET.SubElement(owner, "flickrid")
    flickr_id.text = "Fried Camels"
    name = ET.SubElement(owner, "name")
    name.text = "Jinky the Fruit Bat"
    
    # Add image size
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_info['width'])
    height = ET.SubElement(size, "height")
    height.text = str(image_info['height'])
    depth = ET.SubElement(size, "depth")
    depth.text = "3"  # Assuming RGB images
    
    # Add segmented tag
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"  # VOC format does not typically use segmented field
    
    # Add object annotations
    for obj in annotations:
        obj_elem = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj_elem, "name")
        name.text = obj['category']
        pose = ET.SubElement(obj_elem, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj_elem, "truncated")
        truncated.text = "0"  # Assuming no truncation
        difficult = ET.SubElement(obj_elem, "difficult")
        difficult.text = "0"  # VOC format has no equivalent to 'difficult', so set to 0
        
        # Bounding box
        bndbox = ET.SubElement(obj_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj['bbox'][0])  # xmin
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj['bbox'][1])  # ymin
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj['bbox'][0] + obj['bbox'][2])  # xmin + width -> xmax
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj['bbox'][1] + obj['bbox'][3])  # ymin + height -> ymax
    
    # Convert to a string and save as XML
    tree = ET.ElementTree(annotation)

    # Extract image file extension (supports .jpg, .png, .bmp)
    file_name_without_ext, ext = os.path.splitext(image_info['file_name'])
    
    # Ensure the extension is properly handled for saving XML
    output_path = os.path.join(output_dir, file_name_without_ext + ".xml")
    tree.write(output_path)

def convert_coco_to_voc(coco_annotation_file, output_dir):
    # Read the COCO annotations
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For each image in the dataset
    for image_info in coco_data['images']:
        # Get annotations for the image
        image_id = image_info['id']
        annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                annotations.append({
                    'category': coco_data['categories'][ann['category_id'] - 1]['name'],  # Convert category_id to category name
                    'bbox': ann['bbox']
                })
        
        # Create and save XML
        create_voc_xml(image_info, annotations, output_dir)

train_file = '/your_project_dir/labels.json'  # Adjust as needed
output_dir = '/your_dataset/Annotations'
convert_coco_to_voc(train_file, output_dir)