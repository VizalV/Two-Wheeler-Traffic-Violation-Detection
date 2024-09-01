# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# we use latest Grounding DINO model API that is not official yet
# !git checkout feature/more_compact_inference_api
# !pip install -q -e .
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import cv2
from typing import Dict, List, Optional, Tuple
from supervision.detection.core import Detections
from xml.etree.ElementTree import Element, SubElement
from xml.dom.minidom import parseString
from defusedxml.ElementTree import fromstring, parse, tostring
from groundingdino.util.inference import Model
from typing import List

# image_source, image = load_image(IMAGE_PATH)

# boxes, logits, phrases = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )

# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# plt.imshow(annotated_frame)


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def list_image_files(directory: str) -> List[str]:
    image_extensions = [".jpeg", ".jpg", ".png", ".bmp", ".gif"]
    image_files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in image_extensions]
    return image_files


def save_voc_xml(xml_string: str, file_path: str):
    with open(file_path, 'w') as f:
        f.write(xml_string)


def image_name_to_xml_name(image_name: str) -> str:
    base_name, _ = os.path.splitext(image_name)
    xml_name = f"{base_name}.xml"
    return xml_name

def object_to_pascal_voc(
    xyxy: np.ndarray, name: str, polygon: Optional[np.ndarray] = None
) -> Element:
    root = Element("object")

    object_name = SubElement(root, "name")
    object_name.text = name

    # https://github.com/roboflow/supervision/issues/144
    xyxy += 1

    bndbox = SubElement(root, "bndbox")
    xmin = SubElement(bndbox, "xmin")
    xmin.text = str(int(xyxy[0]))
    ymin = SubElement(bndbox, "ymin")
    ymin.text = str(int(xyxy[1]))
    xmax = SubElement(bndbox, "xmax")
    xmax.text = str(int(xyxy[2]))
    ymax = SubElement(bndbox, "ymax")
    ymax.text = str(int(xyxy[3]))

    if polygon is not None:
        # https://github.com/roboflow/supervision/issues/144
        polygon += 1
        object_polygon = SubElement(root, "polygon")
        for index, point in enumerate(polygon, start=1):
            x_coordinate, y_coordinate = point
            x = SubElement(object_polygon, f"x{index}")
            x.text = str(x_coordinate)
            y = SubElement(object_polygon, f"y{index}")
            y.text = str(y_coordinate)

    return root


def detections_to_pascal_voc(
    detections: Detections,
    classes: List[str],
    filename: str,
    image_shape: Tuple[int, int, int],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> str:

    height, width, depth = image_shape
    # Create root element
    annotation = Element("annotation")

    # Add folder element
    folder = SubElement(annotation, "folder")
    folder.text = "VOC"

    # Add filename element
    file_name = SubElement(annotation, "filename")
    file_name.text = filename

    # Add source element
    source = SubElement(annotation, "source")
    database = SubElement(source, "database")
    database.text = "roboflow.ai"

    # Add size element
    size = SubElement(annotation, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    d = SubElement(size, "depth")
    d.text = str(depth)

    # Add segmented element
    segmented = SubElement(annotation, "segmented")
    segmented.text = "0"
    count=0
    # Add object elements
    for xyxy, _, class_id, _ in detections:
        name = classes[class_id]
        if name=="helmet":
            count+=1
        next_object = object_to_pascal_voc(xyxy=xyxy, name=name)
        annotation.append(next_object)
        
    # Generate XML string
    xml_string = parseString(tostring(annotation)).toprettyxml(indent="  ")

    return xml_string,count
def det_voc_xml(SOURCE_DIRECTORY_PATH, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    helmets=0
    for image_name in list_image_files(SOURCE_DIRECTORY_PATH):
        image_path = os.path.join(SOURCE_DIRECTORY_PATH, image_name)
        image = cv2.imread(image_path)
        height, width, depth = image.shape
        xml_name = image_name_to_xml_name(image_name=image_name)
        xml_path = os.path.join(output_directory, xml_name)

        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        # print(detections)
        # drop potential detections with phrase that is not part of CLASSES set
        detections = detections[detections.class_id != None]
        # drop potential detections with area close to area of whole image
        detections = detections[(detections.area / (height * width)) < 0.9 ]
        # drop potential double detections
        # detections = detections.with_nms()  # Uncomment this line to apply non-maximum suppression
        xml_string,count= detections_to_pascal_voc(
            detections=detections,
            classes=CLASSES,
            filename=image_name, 
            image_shape=image.shape
        )
        helmets+=count
        # print(xml_string)
        save_voc_xml(xml_string=xml_string, file_path=xml_path)
    print(helmets)
    
if __name__ == "__main__":
    CONFIG_PATH = "/home/vishal.v/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join("/home/vishal.v/work/TW_Detection", WEIGHTS_NAME)
    source='/ssd_scratch/datasets/TW_Detection/images/train/*.jpg'
    model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
    CLASSES = ["helmet","number plate"]
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.3

    tr_source = "/ssd_scratch/datasets/TW_Detection/images/train"
    tr_output_directory = "/home/vishal.v/work/auto/train"
    val_source = "/ssd_scratch/datasets/TW_Detection/images/val"
    val_output_directory = "/home/vishal.v/work/auto/val"

    det_voc_xml(tr_source, tr_output_directory)
    det_voc_xml(val_source, val_output_directory)