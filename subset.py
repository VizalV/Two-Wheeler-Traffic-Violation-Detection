import numpy as np 
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import pybboxes as pbx
device = "cuda" if torch.cuda.is_available() else "cpu"
PATH=Path("/ssd_scratch/datasets")
DATASET=PATH/"IDD_Detection"
IMAGES=DATASET/"JPEGImages"
ANNOTATIONS=DATASET/"Annotations"
an_paths=[]
im_paths=[]
for folder_name in os.listdir(IMAGES):
    view_path = os.path.join(IMAGES, folder_name)
    an_view_path = os.path.join(ANNOTATIONS, folder_name)
    for img_folder in os.listdir(view_path):
        img_path = os.path.join(view_path, img_folder)
        an_path = os.path.join(an_view_path, img_folder)
        for img_filename in os.listdir(img_path):
            im_paths.append(os.path.join(img_path,img_filename))
            am_filename = img_filename.replace('.jpg', '.xml')
            an_paths.append(os.path.join(an_path, am_filename))
            
def filter_test(image_paths, annotation_paths):
    filtered_image_paths = []
    filtered_annotation_paths = []

    for img_path, ann_path in zip(image_paths, annotation_paths):
        if os.path.exists(ann_path):
            filtered_image_paths.append(img_path)
            filtered_annotation_paths.append(ann_path)

    return filtered_image_paths, filtered_annotation_paths
im_paths, an_paths = filter_test(im_paths, an_paths)


def count_motorcycles_in_annotation(xml_file):
    count = 0

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall(".//object"):
        name = obj.find("name")
        if name is not None and name.text == "motorcycle":
            count += 1
            break

    return count


motor_cycles=0
mot_im_paths=[]
mot_an_paths=[]
for xml_file in an_paths:
    motor_cycles+=count_motorcycles_in_annotation(xml_file)
    if (count_motorcycles_in_annotation(xml_file)==1):
        mot_an_paths.append(xml_file)
        mot_im_filename = xml_file.replace('.xml', '.jpg').replace('Annotations','JPEGImages')
        mot_im_paths.append(mot_im_filename)
    # print(count_motorcycles_in_annotation(xml_file))
# motor_cycles=count_motorcycles_in_annotation(an_paths[2])
print(f'Total number of annotations with "motorcycle" tag: {motor_cycles}')

with open(DATASET/'train.txt', 'r') as file:
    train_paths = [line.strip() for line in file]

with open(DATASET/'test.txt', 'r') as file:
    test_paths = [line.strip() for line in file]

with open(DATASET/'val.txt', 'r') as file:
    val_paths = [line.strip() for line in file]
os.path.splitext(mot_im_paths[3])[0].split('/JPEGImages/')
motor_ids = [os.path.splitext(path)[0].split('JPEGImages/')[1]  for path in mot_im_paths]
new_train_paths = [path for path in train_paths if path in motor_ids]
new_val_paths = [path for path in val_paths if path in motor_ids]
new_test_paths = [path for path in test_paths if path in motor_ids]

source_root = PATH/'IDD_Detection'
destination_root = PATH/'TW_Detection'

train_image_list = [os.path.join(source_root, 'JPEGImages', path + '.jpg') for path in new_train_paths]
train_annotation_list = [os.path.join(source_root, 'Annotations', path + '.xml') for path in new_train_paths]
val_image_list = [os.path.join(source_root, 'JPEGImages', path + '.jpg') for path in new_val_paths]
val_annotation_list = [os.path.join(source_root, 'Annotations', path + '.xml') for path in new_val_paths]

class_mapping = {
    'motorcycle': 0,
    'rider': 1,
    'helmet':2,
    'number plate': 3
    # 'traffic sign': 4,
    # 'traffic light': 5,
}

def names_to_ids(class_name):
    return class_mapping[class_name]
needed=['motorcycle','rider','helmet','number plate']
def extract_boxes(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []

    for box in root.findall('.//object'):
        name = box.find('name').text   
        if name in needed:
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            width = int(root.find('.//size/width').text)
            height = int(root.find('.//size/height').text)
            # Check if the bounding box has non-zero height and width
            if xmax > xmin and ymax > ymin and xmin<width and ymin<height:
                if width<=xmax:
                    xmax-=(xmax-width+1)
                if height<=ymax:
                    ymax-=(ymax-height+1)
                if ymin<=0:
                    ymin=1
                if xmin<=0:
                    xmin=1
                
                coors = [names_to_ids(name),xmin, ymin, xmax, ymax]
                voc_bbox = coors[1:5]
                W, H = width, height  # WxH of the image
                coors[1:5]=pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(W,H))
                boxes.append(coors)
        # else:
            # print(xmin,xmax,ymin,ymax,width,height)
        
    return boxes

train_boxes=[]
for path in (train_annotation_list):
    train_boxes.append(extract_boxes(path))
val_boxes=[]
for path in (val_annotation_list):
    val_boxes.append(extract_boxes(path))

def sub_dataset(image_list, annotation_list, boxes, destination_root, subset):
    for annotation_file, box in zip(annotation_list, boxes):
        an_path=os.path.splitext(annotation_file)[0].split('Annotations/')[1]
        folder_name = os.path.basename(os.path.dirname(an_path))
        filename=os.path.basename(an_path)
        new_folder = os.path.join(destination_root, "labels",subset)
        os.makedirs(new_folder, exist_ok=True)
        txt_filename = os.path.join(new_folder, folder_name + '_' + filename + '.txt')
        with open(txt_filename, 'w') as txt_file:
            for box_values in box:
                # Convert each value to string and write to the file
                line = ' '.join(map(str, box_values)) + '\n'
                txt_file.write(line)
    
    for image_path in image_list:
        img_sub=os.path.splitext(image_path)[0].split('JPEGImages/')[1]
        folder_name = os.path.basename(os.path.dirname(img_sub))
        filename=os.path.basename(img_sub)
        new_folder = os.path.join(destination_root,"images", subset)
        new_filename = os.path.join(new_folder, folder_name + '_' + filename + '.jpg')
        os.makedirs(new_folder, exist_ok=True)
        # Copy the image to the destination folder
        shutil.copy(image_path, new_filename)          

sub_dataset(train_image_list,train_annotation_list,train_boxes,destination_root,"train")
sub_dataset(val_image_list,val_annotation_list,val_boxes,destination_root,"val")
