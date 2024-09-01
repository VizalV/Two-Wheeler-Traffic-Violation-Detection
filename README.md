
# Two Wheeler Traffic Violation Detection
The above repository gives a complete guide for detecting two wheeler traffic violations using the IDD-Detection Dataset.



## Installation
Install the IDD-Detection dataset from 
```https
https://idd.insaan.iiit.ac.in/dataset/download/
```
Install the requirements using

```bash
  pip install -r requirements.txt
```

## Steps
The pipeline of creating the TW_Dataset from IDD_Dataset is as follows:
+ Run `subset.py` to create TW_Dataset consisting of only 2 wheeler images.
+ The labels are also changed to **YOLO** format. 
+ Run `gdino.py` to create auto annotations of custom classes (helmet, number plate) on the new TW_Dataset.
+ Run  `vocyolo.py` to append annotations to the existing annotations.
+ The sub-dataset is ready to use.

## Violation Detection
+ The fine-tuned YOLOv8 model from `yolo_Train.py` is utilized to identify Motorcycle-Rider
pairs which are cropped and passed to another instance of the model for the
detection of violations  `yolo_crop.py`.
+ A tracking system that links detected violations to unique rider IDs
is then implemented using DeepSORT.
+ The current system can detect violations such as no helmet, triple
riding along with the number plate of the motorcycle.

## RCNN
+ `rcnn` is another model capable of detecting all  15 classes present in the original IDD-Detection Dataset.
+ It can be used to provide the test data necessary for two wheeler traffic violation detection.

## Demo
![](https://github.com/VizalV/TW_Detection/blob/main/Demo.gif)

## Resources

 - [Grounding Dino Repo](https://github.com/IDEA-Research/GroundingDINO)
 - [Grounding Dino Paper](https://arxiv.org/abs/2303.05499)
 - [YOLO](https://github.com/ultralytics/ultralytics)
 - [mAP](https://www.v7labs.com/blog/mean-average-precision)

