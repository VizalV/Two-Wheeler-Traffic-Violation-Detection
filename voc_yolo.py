#%%
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pybboxes as pbx
import shutil
import cv2

#%%
PATH=Path("/ssd_scratch/datasets")
DATASET=PATH/"TW_Detection"
#%%
AUTO_val=Path("/home/vishal.v/work/auto/val")
AUTO_train=Path("/home/vishal.v/work/auto/train")
auto_txmls=os.listdir(AUTO_train)
auto_xmls=os.listdir(AUTO_val)
len(auto_xmls)

# print(auto_xmls[1])
DATASET_val=DATASET/"labels"/"val"
DATASET_train=DATASET/"labels"/"train"
#%%
#%%

class_mapping = {
    'motorcycle': 0,
    'rider': 1,
    'helmet':2,
    'number plate': 3,
    'traffic sign': 4,
    'traffic light': 5,
}

def names_to_ids(class_name):
    return class_mapping[class_name]
#%%
needed=['motorcycle','rider','traffic sign','traffic light','helmet','number plate']
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

#%%
# destination_dir=DATASET/"labels"
#%%
auto_boxes=[]
for path in auto_xmls:
    # print(AUTO_val/path)
    auto_boxes.append(extract_boxes(AUTO_val/path))
#%%
auto_tboxes=[]
for path in auto_txmls:
    auto_tboxes.append(extract_boxes(AUTO_train/path))
len(auto_tboxes)
#%%
count=0
for box in auto_boxes:
    if box==[]:
        count+=1
# count
#%%
# len(auto_boxes)
#%%
def append_txt_annotate(annotation_files,all_boxes,destination_dir):
    for annotation_file, box_data in zip(annotation_files, all_boxes):
        txt_filename = os.path.join( destination_dir,annotation_file)
        txt_filename=txt_filename.replace('.xml','.txt')
        with open(txt_filename, 'a') as txt_file:
            for box_values in box_data:
                # Convert each value to string and write to the file
                line = ' '.join(map(str, box_values)) + '\n'
                txt_file.write(line)

append_txt_annotate(auto_xmls,auto_boxes,DATASET_val)
append_txt_annotate(auto_txmls,auto_tboxes,DATASET_train)
#%%
# def txt_annotate(annotation_files,all_boxes,destination_dir):
#     for annotation_file, box_data in zip(annotation_files, all_boxes):
#         folder_name = os.path.basename(os.path.dirname(annotation_file))
#         subset=annotation_file.split('/')[0]
#         filename=os.path.basename(annotation_file)
#         new_folder = os.path.join(destination_dir, subset)
#         os.makedirs(new_folder, exist_ok=True)
#         txt_filename = os.path.join(new_folder, folder_name + '_' + filename + '.txt')
#         with open(txt_filename, 'w') as txt_file:
#             for box_values in box_data:
#                 # Convert each value to string and write to the file
#                 line = ' '.join(map(str, box_values)) + '\n'
#                 txt_file.write(line)

# #%%
# destination_img_dir=DATASET/"images"
# def copy_images_to_folder(image_list,destination_folder):
#     for img in image_list:
#         img_sub=os.path.splitext(img)[0].split('JPEGImages/')[1]
#         folder_name = os.path.basename(os.path.dirname(img_sub))
#         subset=img_sub.split('/')[0]
#         filename=os.path.basename(img_sub)
#         new_folder = os.path.join(destination_folder, subset)
#         new_filename = os.path.join(new_folder, folder_name + '_' + filename + '.jpg')
#         os.makedirs(new_folder, exist_ok=True)
#         # Copy the image to the destination folder
#         shutil.copy(img, new_filename)               
# %%
# txt_annotate(an_ids,all_boxes,destination_dir)
# copy_images_to_folder(im_paths,destination_img_dir)


# %%

# %%
