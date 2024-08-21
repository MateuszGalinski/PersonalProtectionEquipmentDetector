# PPE Detection Model Training

This repository contains the code for training and deploying a Personal Protective Equipment (PPE) detection model using the YOLOv8 architecture. The model is trained to detect various PPE items such as goggles, helmets, and vests in real-time images and video streams.

## Table of Contents
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running Inference on Images](#running-inference-on-images)
- [Real-Time Detection with Webcam](#real-time-detection-with-webcam)

## Installation

Before running the code, ensure that you have the following dependencies installed:

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- Ultralytics YOLOv8

You can install the required Python packages using pip and requirements file:

pip install -r requirements.txt

## Training

Training was done on a roboflow dataset credited in README.dataset.txt using following code:

```
import torch
import os
from ultralytics import YOLO

# Training
model = YOLO("yolov8n.yaml")
results = model.train(data=os.path.join("dataset_goggles", "data.yaml"), epochs=100)
```

Model was not trained from scratch but from a pretrained yolov8n model for 100 epochs in train2 file (all parameters are in args.yaml file),  
in the future I want to try to train it for more epochs and compare results

## Inference On Images

For this task I use opencv to display image with bounding boxes as shown below:

```
for result in results:
    # Get the original image
    image = result.orig_img

    # Draw the bounding boxes and labels on the image
    annotated_image = result.plot()  # plot() returns an image with detections drawn

    # Display the image
    cv2.imshow("Detection Results", annotated_image)
    cv2.waitKey(0)  # Press any key to close the image window
    cv2.destroyAllWindows()
```

Model performs decently however tends to overlay many classes on each which could be adjusted using NMS and iou-thresholding parameters

## Real-Time Detection With Webcam

Code for this part is as follows:

```
import cv2

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Here you would process the frame and get the result, e.g.,
    results = best_model.predict(frame)
    # For demonstration purposes, let's assume `result` is already obtained.

    for result in results:
        img = result.plot()  # This method plots bounding boxes on the image

    # Display the resulting frame directly in a window
    cv2.imshow('Webcam Feed', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
```

Model performs really fast, no slowage due to detectin, video is responsive, has trouble with detecting transparent goggles at certain angles

## Known issues

### Issues:
- Trouble detecting transparent goggles at certain angles
- Tends to overlay classes of the same and different types

### Solutions:
- Fine tuning with transparent goggles dataset as the current dataset in class goggles contains both transparent and non-transparent goggles making transparent ones less common
- Changing NMS and iou-thresholding to deal with overlapping of classes
