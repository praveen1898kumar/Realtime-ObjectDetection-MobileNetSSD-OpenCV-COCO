# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import matplotlib.pyplot as plt  # Library for plotting images and graphs

# Load the pre-trained model
config_file = '/Users/praveen18kumar/Downloads/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Configuration file for the model
frozen_model = '/Users/praveen18kumar/Downloads/frozen_inference_graph.pb'  # Frozen model file
model = cv2.dnn_DetectionModel(frozen_model, config_file)  # Create a detection model instance

# Load class labels from a text file
classLabels = []
file_name = '/Users/praveen18kumar/Downloads/labels.txt'  # Path to the file containing class labels
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')  # Read and parse class labels

# Configure model input parameters
model.setInputSize(320, 320)  # Set input size for the model
model.setInputScale(1.0/127.5)  # Set input scale for normalizing pixel values
model.setInputMean((127.5, 127.5, 127.5))  # Set input mean for normalization
model.setInputSwapRB(True)  # Set swapRB parameter to True to swap R and B channels

# Webcam capture setup
cap = cv2.VideoCapture(1)  # Open the first webcam (index 1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # If the first webcam is not available, try the default (index 0)
if not cap.isOpened():
    raise IOError("Can't open the video")  # Raise an error if webcam cannot be opened

# Define font properties for displaying labels
font_scale = 3  # Font scale for label text
font = cv2.FONT_HERSHEY_PLAIN  # Font type for label text

# Main loop for real-time object detection
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # If frame cannot be read, exit the loop

    # Perform object detection on the frame
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)  # Detect objects with confidence threshold

    print(ClassIndex)  # Print detected class indices

    # Draw bounding boxes and labels on the frame
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:  # Check if class index is within the expected range
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)  # Draw bounding box
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)  # Add label text

    # Display the frame with object detection results
    cv2.imshow('Object detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
