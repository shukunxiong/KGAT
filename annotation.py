import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

classes_path    = 'KGAT_yolo_v8/model_data/your_data'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent is used to specify the ratio of (training set + validation set) to test set. By default, (training set + validation set) : test set = 9:1
#   train_percent is used to specify the ratio of training set to validation set within (training set + validation set). By default, training set : validation set = 9:1
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9

dataset_path  = '/root/autodl-tmp/filr2_coco'

dataset_sets  = [('train'), ('val')]
classes, _      = get_classes(classes_path)

#-------------------------------------------------------#
#   Count the number of objects
#-------------------------------------------------------#
photo_nums  = np.zeros(len(dataset_sets))
nums        = np.zeros(len(classes))
def convert_annotation_own(image_id, list_file):
    in_file = open(os.path.join(dataset_path, 'Annotations','%s.xml'%(image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(dataset_path):
        raise ValueError("Spaces cannot exist in the folder path where the dataset is stored or in image names, as this will affect normal model training. Please modify accordingly.")
    print("Generate train.txt and val.txt for train.")
    type_index = 0
    for year, image_set in dataset_sets:
        # 这个是用于加载我自己数据集的
        image_ids = open(os.path.join(dataset_path, 'Main','%s.txt'%(image_set)), encoding='utf-8').read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/2017_clean/%s.jpg'%(os.path.abspath(dataset_path), image_id))
            convert_annotation_own(image_id, list_file)
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print("Generate train.txt and val.txt for train done.")
    
    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()

    str_nums = [str(int(x)) for x in nums]
    tableData = [
        classes, str_nums
    ]
    colWidths = [0]*len(tableData)
    len1 = 0
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                colWidths[i] = len(tableData[i][j])
    printTable(tableData, colWidths)

    if photo_nums[0] <= 500:
        print("The number of training samples is less than 500, which is a relatively small dataset size. Please note to set a larger number of training epochs to ensure sufficient gradient descent steps.")

    if np.sum(nums) == 0:
        print("No objects were detected in the dataset. Please note to modify classes_path to match your own dataset and ensure that the label names are correct; otherwise, training will have no effect!")