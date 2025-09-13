import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
import random
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO
from attack import *
from robust import *
from nets.yolo import YoloBody
from utils.utils import get_classes
from utils.utils import cvtColor
from torchvision import transforms

DATASET_MEANS = (104, 117, 123)
def preprocess_input(image):
    image /= 255.0
    return image



def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

# Salt Noise
def apply_salt_and_pepper_noise(image, prob):
    image = np.array(image)
    
    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Randomly generate a value between [0, 1)
            random_value = random.random()

            # If less than prob, flip the pixel (salt or pepper noise)
            if random_value < prob:
                # Randomly decide between salt noise (white) or pepper noise (black)）
                image[i, j] = 255 if random.random() < 0.5 else 0

    return Image.fromarray(image)


# Gaussian Noise
def apply_gaussian_noise(image, var):
    image = np.array(image)
    mean = 0
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian, 0, 255)  # Ensure pixel values do not exceed 255
    return Image.fromarray(noisy_image.astype(np.uint8))

# Main function for adding disturbances
def add_disturbance(image_tensor, blur_radius=0, salt_prob=0, gaussian_var=0):
    # Convert input image from tensor to PIL image上
    image = image_tensor.squeeze().cpu().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    image = Image.fromarray(image.astype(np.uint8))

    # Apply Gaussian blur
    if blur_radius > 0:
        image = apply_gaussian_blur(image, blur_radius)

    # Apply Salt noise
    if salt_prob > 0:
        image = apply_salt_and_pepper_noise(image, salt_prob)

    # Apply Gaussian noise
    if gaussian_var > 0:
        image = apply_gaussian_noise(image, gaussian_var)

    return image

def get_revised_local(bndbox,scale_w,scale_h,dx,dy,input_shape):
    left = float(bndbox.find('xmin').text) # Use float() instead of int()
    top = float(bndbox.find('ymin').text)   
    right = float(bndbox.find('xmax').text) 
    bottom = float(bndbox.find('ymax').text)

    # Scale the labels accordingly
    left = left*scale_w+dx
    top  = top*scale_h+dy
    right = right*scale_w+dx
    bottom = bottom*scale_h+dy
    
    # Convert floating-point coordinates to integers (if needed)
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

def load_image_and_labels(image_path, annotation_path, input_shape,class_file):
    """
    加载图像和对应的标签，返回符合labels_out格式的标签。
    :param image_path: 图像的路径
    :param annotation_path: 对应的VOC格式标注文件路径
    :param input_shape: 输入图像的尺寸（宽，高），用于归一化
    :return: image, labels_out
    """
    # 加载类别
    class_names = load_classes(class_file)
    # 加载图像
    old_image = Image.open(image_path)
    old_image   = cvtColor(old_image)
    iw, ih = old_image.size  # 获取原始图像的宽和高
    h, w = input_shape    # 获取目标图像尺寸
    # 获取在目标尺寸下的缩放系数
    scale = min(w/iw, h/ih)
    # 得到保证图片在不失真的情况下缩放的比例
    nw = int(iw*scale)
    nh = int(ih*scale)
    # 不足的部分补充灰条，这里是获取灰条的大小
    dx = (w-nw)//2
    dy = (h-nh)//2
    old_image       = old_image.resize((nw,nh), Image.BICUBIC)
    new_image   = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(old_image, (dx, dy))
    image = np.array(new_image)  # 转换为 numpy 数组
    image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
    
    # 获取标注信息
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # 存储所有的标签信息
    labels_out = []

    for obj in root.findall('object'):
        # 获取标签的基本信息
        obj_name = obj.find('name').text
        difficult_flag = False
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                difficult_flag = True
        
        # 获取边界框
        bndbox = obj.find('bndbox')
        left = float(bndbox.find('xmin').text)  # 使用 float() 而不是 int()
        top = float(bndbox.find('ymin').text)   # 使用 float() 而不是 int()
        right = float(bndbox.find('xmax').text) # 使用 float() 而不是 int()
        bottom = float(bndbox.find('ymax').text)# 使用 float() 而不是 int()
        
        # 将坐标进行调整
        box = [left *nw/iw + dx, top *nh/ih + dy, right *nw/iw + dx, bottom *nh/ih + dy]
        
        # 将浮动的坐标值转换为整数（如果需要）
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        
        scale_h = nh/ih
        scale_w = nw/iw
        # 进一步裁剪，保证边界框的合法性
        box[0] = np.clip(box[0],0, w)
        box[1] = np.clip(box[1],0, h)
        box[2] = np.clip(box[2],0, w)
        box[3] = np.clip(box[3],0, h)
        # 计算真实框的中心坐标和宽高
        box[2] = box[2] - box[0]  # 宽度
        box[3] = box[3] - box[1]  # 高度
        box[0] = box[0] + box[2] / 2  # 中心X
        box[1] = box[1] + box[3] / 2  # 中心Y
        # 将具体的物品名转化为数字
        if obj_name in class_names:
            class_id = class_names[obj_name]  # 根据类别名找到对应的类别ID
        else:
            print(f"Warning: '{obj_name}' not found in class_names.")
            print(f"Current obj_name: '{obj_name}'")
            print(f"Available class names: {list(class_names.keys())}")
            continue  # 如果类别不在类别文件中则跳过该物体
        
        # 存储标签信息
        labels_out.append([0, class_id, *box])  # 格式为[标签ID, 类别, x_center, y_center, width, height]
    
    # 转换为numpy数组
    labels_out = np.array(labels_out, dtype=np.float32)
    image = torch.tensor(image).float().to('cuda')
    image = image.unsqueeze(0) 
    labels_out = torch.tensor(labels_out, dtype=torch.float32).to('cuda')
    return image, labels_out,scale_w,scale_h,dx,dy

# This part is used to load images and their corresponding labels, and finally generate adversarial examples
def load_classes(class_file):
    """
    Load images and their corresponding labels, and return labels in the format of labels_out.
    :param class_file:  Path to the text file containing class names
    :return: Dictionary mapping class names to their indices
    """
    class_names = {}
    with open(class_file, 'r') as f:
        for idx, line in enumerate(f):
            class_names[line.strip()] = idx  
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

    # The size of the input image must be a multiple of 32
    input_shape       = [640, 640]
    local_rank      = 0
    #--------------------------------------------------------------------------------------#
    #   classes_path is used to specify the classes for which VOC_map needs to be calculated.
    #   Generally, it should be consistent with the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/M3FD_classes.txt'
    # Specify the model type. This needs to be specified once in yolo.py and also here.
    phi                     = 'x'
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MINOVERLAP      = 0.5
     #-------------------------------------------------------------------------------------------------------------------#
    #   Due to the principle of mAP calculation, the network needs to obtain nearly all prediction boxes to calculate mAP.
    #   Therefore, the confidence value should be set as small as possible to obtain all possible prediction boxes.
    #   
    #   This value is generally not adjusted. Because calculating mAP requires obtaining nearly all prediction boxes,
    #   the confidence defined here cannot be changed arbitrarily.
    #   To obtain Recall and Precision values under different thresholds, modify the score_threhold below.
    #--------------------------------------------------------------------------------------------------------------------#
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
    dataset_path  = 'your_dataser/'
    #-------------------------------------------------------#
    #   Folder for outputting results, default is map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(dataset_path , "/root/autodl-tmp/M3FD_coco_new/Main/val.txt")).read().strip().split()
    
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))
    class_names, num_classes = get_classes(classes_path)
    model = YoloBody(input_shape, num_classes, phi, pretrained=False)
    model = model.to('cuda')

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
           
            #------------------------add common corruptions-----------------------------------------------#
            # The following operations are to ensure proper processing by Image
            # Add noise here.blur_radius.
            # blur_radius, salt_prob, and gaussian_var represent adding Gaussian blur,
            # salt and pepper noise, and Gaussian noise respectively.
            # Noise additions can be superimposed.
            # If a certain type of noise is not desired, set its corresponding value to 0.
            image_PIL = add_disturbance(image_clean, blur_radius=0, salt_prob=0, gaussian_var=0)
            #---------------------------------------------------------------------------------------------#

            transform = transforms.Compose([
                transforms.Resize((640, 640)),      #  # Ensure the image size is 640x640
                transforms.ToTensor(),              # Convert to tensor in the range [0, 1]])
                            ])

            # 2. Convert the PIL image to a PyTorch tensor and add a batch dimension
            image_tensor = transform(image_PIL)  # Convert to shape (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)  #  Add batch dimension to get shape (1, C, H, W)
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
                root = ET.parse(os.path.join(dataset_path , "Annotations/"+image_id+".xml")).getroot()
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