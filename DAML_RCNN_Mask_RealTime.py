# Driver Assist Machine Learning Using Masking-RCNN

import cv2
import numpy as np
import os
import sys
from soco import coco
from mrcnn import utils
from mrcnn import model as modellib


# This portion of the code is specifying the path to the appropiate directories, while also grabbing the weights for the pre-trained model.
# The mask_rcnn_coco.h5 file is a pre-trained dataset provided by matterport that act as weights for MS COCO. It is mask-RCNN trained 
# for object detection.
dirMain = os.path.abspath("./")
dirModel = os.path.join(dirMain, "logs")
sys.path.append(os.path.join(dirMain,"/coco/"))
path_Coco = os.path.join(dirMain, "mrcnn/mask_rcnn_coco.h5")


# A configuration object is required to make an inference for the Mask_RCNN instance 
# The configuration is set to specify the number of images per batch
class Configure_coco(coco.CocoConfig):
    # Since we are running inference 1 image at a time, batch size is set to 1. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


    
# Creating an object of class Configure_coco to configure the masking model
nConfig = Configure_coco()
nConfig.display()

# MaskRCNN instance object created in inference mode since this mode is used to make estimations for a given image, the dirModel variable is the 
# path to where the log messages will be stored
mrcnn_model = modellib.MaskRCNN(
    mode="inference", model_dir=dirModel, config=nConfig
)

# Load the weights that will be used to calculate the estimations, and assist in classifying the detected object in the frame
mrcnn_model.load_weights(path_Coco, by_name=True)

# Classification types to compare to for the given trained model 
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# This function applies a cyan coloured mask with a 50% opacity to the ROI detected in the source image
def apply_mask(cyan_col, mask, source, transp=0.5):
    for n, c in enumerate(cyan_col):
        source[:, :, n] = np.where(
            mask == 1,
            source[:, :, n] * (1 - transp) + transp * c,
            source[:, :, n]
        )
    return source



# Apply the mask, bounding box, and classification to the region of interest
def mask_frame(source, region_interest, masks, class_ids, cls_names, scores):
    # Number of instances found in ROI
    n_instances = region_interest.shape[0]
    if not n_instances:
        print('NO Instances FOUND in ROI')
    else:
        assert region_interest.shape[0] == masks.shape[-1] == class_ids.shape[0]
    # For each instance found apply mask, box, and label
    for i in range(n_instances):
        # Detect only road obstacles from the class names specified in the class_names array above. class_names[1 .. 14]
        if class_ids[i] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            if not np.any(region_interest[i]):
                continue
            
            # Coordinates for region of interest
            y1, x1, y2, x2 = region_interest[i]
            # Classification for the ROI
            label = class_names[class_ids[i]]
            # Confidence score in relation to its classification
            score = scores[i] if scores is not None else None
            # Store classification and score as a string caption to the object detected to be used as a label
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]
            
            # Cyan color for mask / bounding box / label in BGR  
            cyan_col = (240,252,3)
            # Apply the mask on the detected object
            source = apply_mask(cyan_col, mask, source)
            # Draw bounding box using the x/y coordinates from the roi on the detected object
            source = cv2.rectangle(source, (x1, y1), (x2, y2), cyan_col, 1)
            # Write the label classification above the detected object using the x/y coordinates
            source = cv2.putText(
                source, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, cyan_col, 1
            )
    return source


    
# Capture Video Real-Time from Camera:
stream = cv2.VideoCapture(0)

# get video capture size 
width  = stream.get(cv2.CAP_PROP_FRAME_WIDTH)  # float value, converted to integer in the next line when writing
height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT) # float value, converted to integer in the next line when writing

# Create VideoWriter object
# 0x7634706d  is the (*'MP4V') video writing formatting, with an output resolution of the original size.
video_output = cv2.VideoWriter('OutputVideo/output.mp4', 0x7634706d, 60.0, (int(width),int(height)))
   
# Start capturing footage frame by frame and apply mask
while True:
    # read in the stream wether its live camera feed or a video footage
    is_streaming , frame = stream.read()
    if not is_streaming:
        print("Finished stream, ending program")
        break
    #Make a prediction with the model creating a dictionary with a set of key value pairs that list possible objects detected 
    get_frame_results = mrcnn_model.detect([frame], verbose=1)

    # Apply the bounding boxes, mask and classification to the footage after setting up the dictionary of key value pairs 
    # Following keypoints in the dictionary
    # rois: Bounding boxes / regions of interest (ROI) for objects detected
    # masks: Masks to generate for objects detected 
    # class_ids: Reference to the classification types
    # scores: Confidence score in relation to its classification to determine what it is
    detected_frame = get_frame_results[0]
    masked_image = mask_frame(frame, detected_frame['rois'], detected_frame['masks'], detected_frame['class_ids'], 
                            class_names, detected_frame['scores'])
    # Write to the video output
    video_output.write(masked_image)
    cv2.imshow("Driver Assist Machine Learning RCNN Mask",masked_image)
    # Press 'q' to exit the program early, the output video file will still be generated if terminated early
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
# Release Stream and video writing
stream.release()
video_output.release()
cv2.destroyWindow("Driver Assist Machine Learning RCNN Mask")
