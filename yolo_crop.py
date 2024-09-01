
import datetime
import random
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort

# CROPPING MOTORCYCLE-RIDER PAIRS
def crop_motor(model, img):
    unique_set=set()
    # img = cv2.imread(img)
    result = model(img, imgsz=640,conf=0.5,agnostic_nms=True)
    res=[]
    # Filter boxes
    class_0_boxes = []
    class_1_boxes = []
    for i in range(len(result[0])):
        if result[0][i].boxes.cls == 0:
            class_0_boxes.append((result[0][i].boxes.xyxy.tolist(),result[0][i].boxes.conf))
        elif result[0][i].boxes.cls == 1:
            class_1_boxes.append(result[0][i].boxes.xyxy.tolist())
    for box_0,conf in class_0_boxes:
        for box_1 in class_1_boxes:
            if box_0 and box_1:
                if overlap(box_0, box_1):
                    x1,x2,y1,y2 = crop_outermost(img, box_0, box_1)
                    if all(coord not in unique_set for coord in [x1, x2, y1, y2]):
                        unique_set.update([x1,y1,x2,y2])
                        res.append([[x1, y1, x2-x1, y2-y1],conf])
    # print(len(res))
    # res = filter_close_boxes(res, min_distance=150)
    # print(len(res))
    return res

def overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1[0]
    x1_2, y1_2, x2_2, y2_2 = box2[0]

    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

def crop_outermost(image, box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1[0]
    x1_2, y1_2, x2_2, y2_2 = box2[0]

    x1_crop = min(x1_1, x1_2)
    y1_crop = min(y1_1, y1_2)
    x2_crop = max(x2_1, x2_2)
    y2_crop = max(y2_1, y2_2)

    return x1_crop, x2_crop, y1_crop, y2_crop

# VISUALIZATION
def plot_images_in_grid(images, rows, cols, violations):
    # Create a new figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

    # Loop through the images and plot them on the grid
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if cols==len(images):
                axes[j].imshow(images[index])
                axes[j].axis('off')
                axes[j].set_title(violations[index])
            elif index < len(images):
                axes[i,j].imshow(images[index])
                axes[i,j].axis('off')
                axes[i,j].set_title(violations[index])
    plt.show()

# USE THE SAME MODEL TO DETECT HELMET, NUMBER PLATES ON THE CROPPED IMAGES (FOR BETTER RESULTS)
def model_check(model, image, imgsz, conf):
    classes=[]
    plates=[]
    result = model(image, imgsz=imgsz, conf=conf,agnostic_nms=True)
    for i in range(len(result[0])):
        classes.append(result[0][i].boxes.cls.to('cpu').numpy())
        if result[0][i].boxes.cls == 5:
            plates.append(result[0][i].boxes.xyxy.tolist())
    # print(classes)
    return classes,plates


# DETERMINING VIOLATIONS FROM THE DETECTIONS
def detect(model,img,boxes):
    violations=[]
    plates=[]
    for i,box in enumerate(boxes):
        riders=0
        motor=0
        image=img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # Edge cases
        if image is None or image.shape[0]==0 or image.shape[1]==0:
            violations.append(" ")
            continue
        if box[2]-box[0]<100 or box[3]-box[1]<100:
            violations.append(" ")
            continue
        # Get plate dimensions from the cropped detections
        classes,plate=model_check(model, image,320,0.1)
        for x in classes:
            if x==1:
                riders+=1
            if x==0:
                motor+=1
        if len(plate)>0:
            pl=plate[0]
            # print(pl)
            # Map the plate dimensions back to the original image dimensions
            for pl_box in pl:
                pl_box[0]=pl_box[0]+boxes[i][0]
                pl_box[1]=pl_box[1]+boxes[i][1]
                pl_box[2]=pl_box[2]+boxes[i][0]
                pl_box[3]=pl_box[3]+boxes[i][1]
            plates.append(pl[0])      
        # classes=model_check(model, image,320,0.1)
        if riders>=(motor+4):
            violations.append(f"Triple riding")
        elif 4 in classes:
            violations.append(" ")
        else:
            violations.append(f"No helmet")
    return violations,plates


def track_and_detect(model, img,color):
    # Get cropped images
    res= crop_motor(model, img)
    # Initialize DeepSORT tracker
    tracks = tracker.update_tracks(res, frame=img)
    boxes=[]
    track_ids=[]
    annotator = Annotator(img)
    for track in tracks:
        # print("i-",i)
        if not track.is_confirmed():
            continue
        track_id=track.track_id
        # left, top, right, bottom format for box
        ltrb = track.to_ltrb(orig_strict=True)
        box=ltrb
        boxes.append(box)
        track_ids.append(track_id)
    # For Violations and plates for each cropped box in a particular frame
    violations,plates=detect(model, img, boxes)
    for i,box in enumerate(boxes):
        # Annotate the boxes with the violations
        c=f"#{track_ids[i]} {violations[i]}"
        annotator.box_label(box, c, color=color[int(track_ids[i])%100])
    if len(plates)>0:
        for plate in plates:
            annotator.box_label(plate, "Number Plate" )
    img = annotator.result()  
    return img


if __name__ == "__main__":
    model = YOLO('runs/detect/train4/weights/best.pt')
    # Age is a hyperparameter that can be played with
    tracker = DeepSort(max_age=15)
    color=[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(100)]
    # Name of video with the real time violation detections
    video_name = 'videos/custom_video.avi'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (1920,1080))  
    # line_pts = [(0, 360), (1280, 360)]
    # Name for the video to be tested
    video_path = "videos/testing3.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        start = datetime.datetime.now()
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            frame=track_and_detect(model, frame,color)
            end = datetime.datetime.now()
        # show the time it took to process 1 frame
            print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
            # calculate the frame per second and draw it on the frame
            fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            video.write(frame)
        else:
            break
    
    video.release()
