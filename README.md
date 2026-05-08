# YOLOv8 Object Detection

This notebook demonstrates how to perform object detection using the YOLOv8 model from the `ultralytics` library in a Google Colab environment. It includes steps for setting up the environment, loading a pre-trained model, running inference on sample images, and displaying the results.

## Setup and Model Loading

First, the necessary libraries are installed and imported, and the `yolov8n.pt` (nano version) model is loaded.

```python
!pip install ultralytics
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

model = YOLO('yolov8n.pt') # 'n' = nano, fastest version
Inference on a Sample Image
A sample image (bus.jpg) is downloaded, and object detection is performed. The detected objects are outlined with bounding boxes, and the resulting image is saved and displayed.

Output Image (output.jpg)
This image would show the bus.jpg with detected objects (e.g., persons, bus, stop sign) and their bounding boxes.
<img width="810" height="1080" alt="image" src="https://github.com/user-attachments/assets/d4175844-8dc9-494d-8d54-b3b7d1fc72bb" />

image 1/1 /content/bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 398.8ms
Speed: 16.3ms preprocess, 398.8ms inference, 50.3ms postprocess per image at shape (1, 3, 640, 480)
Inference on an Uploaded Image
The notebook also allows users to upload their own image for object detection. The uploaded image is processed, and the results are saved and displayed.

Output Image (my_output.jpg)
This image would show your uploaded image (TEST@.webp in this example) with detected objects (e.g., persons, cars) and their bounding boxes.
<img width="321" height="180" alt="image" src="https://github.com/user-attachments/assets/09a37d2f-bd3e-4700-aaf1-ec1de3f02a9d" />
Saving TEST@.webp to TEST@.webp


image 1/1 /content/TEST@.webp: 384x640 7 persons, 8 cars, 154.1ms
Speed: 2.7ms preprocess, 154.1ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
Listing Detected Objects and Confidence Scores
Finally, the detected objects and their confidence scores from the last inference (on the uploaded image) are printed to the console.

Detected: car | Confidence: 0.89
Detected: person | Confidence: 0.89
Detected: person | Confidence: 0.73
Detected: person | Confidence: 0.68
Detected: car | Confidence: 0.61
Detected: person | Confidence: 0.59
Detected: car | Confidence: 0.53
Detected: car | Confidence: 0.51
Detected: person | Confidence: 0.49
Detected: person | Confidence: 0.41
Detected: car | Confidence: 0.34
Detected: car | Confidence: 0.33
Detected: person | Confidence: 0.32
Detected: car | Confidence: 0.28
Detected: car | Confidence: 0.28
