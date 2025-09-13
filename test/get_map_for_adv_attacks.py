import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils_map import get_coco_map, get_map
from yolo import YOLO
from attack import *
from robust import *
from nets.yolo import YoloBody
from utils.utils import get_classes
from utils.utils import cvtColor

DATASET_MEANS = (104, 117, 123)
def preprocess_input(image):
    image /= 255.0
    return image

def get_revised_local(bndbox,scale_w,scale_h,dx,dy,input_shape):
    left = float(bndbox.find('xmin').text)  # Use float() instead of int()
    top = float(bndbox.find('ymin').text)   
    right = float(bndbox.find('xmax').text) 
    bottom = float(bndbox.find('ymax').text)

    # Scale the labels accordingly
    left = left*scale_w+dx
    top  = top*scale_h+dy
    right = right*scale_w+dx
    bottom = bottom*scale_h+dy
    
    # Convert floating-point coordinates to integers 
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)
    
    # Further crop to ensure the bounding box is valid
    left = np.clip(left,0,input_shape[0])
    top = np.clip(top,0,input_shape[1])
    right = np.clip(right,0,input_shape[0])
    bottom = np.clip(bottom,0,input_shape[1])
    
    return left,top,right,bottom

# This part is used to load images and their corresponding labels, and finally generate adversarial examples
def load_image_and_labels(image_path, annotation_path, input_shape,class_file):
    """
    Load images and their corresponding labels, and return labels in the format of labels_out.
    :param image_path: Path to the image
    :param annotation_path: Path to the corresponding VOC-format annotation file
    :param input_shape: Size of the input image (width, height) for normalization
    :return: image, labels_out
    """
    # Load classes
    class_names = load_classes(class_file)
    # Load image
    old_image = Image.open(image_path)
    old_image   = cvtColor(old_image)
    iw, ih = old_image.size  # Get the width and height of the original image
    h, w = input_shape    # Get the target image size
    
    # Calculate the scaling factor for the target size
    scale = min(w/iw, h/ih)
    # Get the scaling ratio that keeps the image undistorted
    nw = int(iw*scale)
    nh = int(ih*scale)
    # Calculate the size of the gray bars to be added for padding
    dx = (w-nw)//2
    dy = (h-nh)//2
    old_image       = old_image.resize((nw,nh), Image.BICUBIC)
    new_image   = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(old_image, (dx, dy))
    image = np.array(new_image)  # Convert to numpy array
    image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
    
    # Get annotation information
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Store all label information
    labels_out = []

    for obj in root.findall('object'):
        # Get basic information of the label
        obj_name = obj.find('name').text
        difficult_flag = False
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                difficult_flag = True
        
        # Get the bounding box
        bndbox = obj.find('bndbox')
        left = float(bndbox.find('xmin').text)  # Use float() instead of int(
        top = float(bndbox.find('ymin').text)   
        right = float(bndbox.find('xmax').text) 
        bottom = float(bndbox.find('ymax').text)
        
        # Adjust the coordinates
        box = [left *nw/iw + dx, top *nh/ih + dy, right *nw/iw + dx, bottom *nh/ih + dy]
        
        # Convert floating-point coordinates to integers 
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        
        scale_h = nh/ih
        scale_w = nw/iw
        # Further crop to ensure the bounding box is valid
        box[0] = np.clip(box[0],0, w)
        box[1] = np.clip(box[1],0, h)
        box[2] = np.clip(box[2],0, w)
        box[3] = np.clip(box[3],0, h)
        # Calculate the center coordinates, width, and height of the ground truth box
        box[2] = box[2] - box[0]  # Width
        box[3] = box[3] - box[1]  # Height
        box[0] = box[0] + box[2] / 2  # Center X
        box[1] = box[1] + box[3] / 2  # Center Y
        
        # Convert specific object names to numbers
        if obj_name in class_names:
            class_id = class_names[obj_name]  # Find the corresponding class ID based on the class name
        else:
            print(f"Warning: '{obj_name}' not found in class_names.")
            print(f"Current obj_name: '{obj_name}'")
            print(f"Available class names: {list(class_names.keys())}")
            continue  # Skip the object if its class is not in the class file
        
        # Store label information
        labels_out.append([0, class_id, *box])  # Format: [label ID, class, x_center, y_center, width, height]
    # Convert to numpy array
    labels_out = np.array(labels_out, dtype=np.float32)
    image = torch.tensor(image).float().to('cuda')
    image = image.unsqueeze(0) 
    labels_out = torch.tensor(labels_out, dtype=torch.float32).to('cuda')
    return image, labels_out,scale_w,scale_h,dx,dy

# This is used to map specific class names to line numbers
def load_classes(class_file):
    """
    Load the list of classes from a text file, where each line represents a class name
    :param class_file: Path to the text file containing class names
    :return: Dictionary mapping class names to their indices
    """
    class_names = {}
    with open(class_file, 'r') as f:
        for idx, line in enumerate(f):
            class_names[line.strip()] = idx  # Map class names to their line numbers
    return class_names

if __name__ == "__main__":
    '''
    Unlike AP (which is an area-based metric), Recall and Precision vary with the confidence threshold.
    By default, the Recall and Precision values calculated in this code correspond to a confidence threshold of 0.5.

    Due to the principle of mAP calculation, the network needs to obtain nearly all prediction boxes to calculate
    Recall and Precision values under different threshold conditions. Therefore, the number of boxes in the txt files
    under map_out/detection-results/ generated by this code is generally larger than that obtained by direct prediction.
    The purpose is to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify what the file calculates when running
    #   map_mode = 0: Full mAP calculation process, including obtaining prediction results, ground truth boxes, and calculating VOC_map.
    #   map_mode = 1: Only obtain prediction results.
    #   map_mode = 2: Only obtain ground truth boxes.
    #   map_mode = 3: Only calculate VOC_map.
    #   map_mode = 4: Use the COCO toolbox to calculate the 0.50:0.95 mAP of the current dataset.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #------------------------------------------------------------------------------------#
    #   This part is used to generate adversarial examples. Specify key parameters for generating adversarial examples.
    #   Supported adversarial types include loc, cls, mtd.
    #-----------------------------------------------------------------------------------#
    adv_type = 'mtd'
    eps = 4/255
    alpha = 4/255
    iters = 10
    # The size of the input image must be a multiple of 32
    input_shape       = [640, 640]
    local_rank      = 0
    #--------------------------------------------------------------------------------------#
    #   classes_path is used to specify the classes for which VOC_map needs to be calculated.
    #   Generally, it should be consistent with the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/your_data.txt'
    # Specify the model type. This needs to be specified once in yolo.py and also here.
    phi                     = 'x'
    # Specify the model path for generating adversarial examples
    model_path      = '/root/autodl-tmp/ablation/loss_M3FD_gama_15mtd/best_epoch_weights.pth'
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MINOVERLAP      = 0.5
    #----------------------------------------------------------------------------------------------------------------------#
    #   Due to the principle of mAP calculation, the network needs to obtain nearly all prediction boxes to calculate mAP.
    #   Therefore, the confidence value should be set as small as possible to obtain all possible prediction boxes.
    #   
    #   This value is generally not adjusted. Because calculating mAP requires obtaining nearly all prediction boxes,
    #   the confidence defined here cannot be changed arbitrarily.
    #   To obtain Recall and Precision values under different thresholds, modify the score_threhold below.
    #----------------------------------------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   The value of non-maximum suppression (NMS) used during prediction. Larger values mean less strict NMS.
    #   This value is generally not adjusted.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Unlike AP (which is an area-based metric), Recall and Precision vary with the threshold.
    #   
    #   By default, the Recall and Precision values calculated in this code correspond to a threshold of 0.5
    #   Since calculating mAP requires obtaining nearly all prediction boxes, the confidence defined above cannot be changed arbitrarily.上
    #   A dedicated score_threhold is defined here to represent the threshold, so that the corresponding Recall and
    #   Precision values can be found when calculating mAP.
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis is used to specify whether to enable visualization for calculation
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   Point to the folder where your dataset is located
    #   By default, it points to your dataset in the root directory
    #-------------------------------------------------------#
    dataset_path  = 'your_dataser/'
    #-------------------------------------------------------#
    #   Folder for outputting results, default is map_out
    #-------------------------------------------------------#
    map_out_path    = 'your_flie/KGAT_yolo_v8/map_out'

    image_ids = open(os.path.join(dataset_path, "your_dataser/Main/val.txt")).read().strip().split()
    #-----------下面这个是原本的加载--------------------------------------#
    # image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))
        
    # Create the model for subsequent evaluation
    class_names, num_classes = get_classes(classes_path)
    model = YoloBody(input_shape, num_classes, phi, pretrained=False)
    
    #------------Load the YOLO model for subsequent generation of adversarial examples--------#
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   For weight files, please refer to the README and download from Baidu Netdisk
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
                print(f"{k} success matching")
            else:
                no_load_key.append(k)
                print(f"{k} fali matching")
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Display keys that failed to match
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m it is normal with no Head,but it is abnormal with no backbone\033[0m")
    model = model.to('cuda')
    # Create a loss function for generating adversarial examples
    yolo_adv_loss = Loss_adv(model)
    #-------------Iterator for generating adversarial examples-----------------------------
    dataset_mean_t = torch.tensor(DATASET_MEANS).view(1, -1, 1, 1)
    pgd = PGD(model, img_transform=(lambda x: x - dataset_mean_t, lambda x: x + dataset_mean_t))
    pgd.set_para(eps=eps, alpha=lambda:alpha, iters=iters)
    adv_dict = {'cls': (CLS_ADG, yolo_adv_loss), 'loc': (LOC_ADG, yolo_adv_loss), 'con': (CON_ADG, yolo_adv_loss),
            'mtd': (MTD, yolo_adv_loss)}
    adv_item = adv_dict[adv_type.lower()]
    adv_generator = adv_item[0](pgd, adv_item[1])
    #------------------End of adversarial example iterator---------------------

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            
            #---------Special reminder: Please modify the image suffix to match the corresponding image ！！！
            image_path  = os.path.join(dataset_path, "clean/"+image_id+".png") # You can change the PNG suffix to the suffix of the dataset.
            #-----------------------------------------------------------------------------------------------
            
            annotation_path = os.path.join(dataset_path, "Annotations/"+image_id+".xml")
            # image       = Image.open(image_path)
            image_clean,boxes_clean,scale_w,scale_h,dx,dy = load_image_and_labels(image_path, annotation_path, input_shape,classes_path)
            # Add a batch dimension
            image_tensor = adv_generator(image_clean,boxes_clean,model.train())
            # The following operations are to ensure proper processing by Image
            image =  image_tensor.cpu().numpy()
            image = image.squeeze() 
            image = np.transpose(image, (1, 2, 0))*255
            image =  Image.fromarray(image.astype(np.uint8))
            
            if map_vis:
                # image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".png"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                # root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                root = ET.parse(os.path.join(dataset_path, "Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    # Since the image size needs to be adjusted when generating adversarial examples,
                    # the read bounding boxes also need to be adjusted accordingly
                    bndbox = obj.find('bndbox')
                    left,top,right,bottom = get_revised_local(bndbox,scale_w,scale_h,dx,dy,input_shape)
                
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, False, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
