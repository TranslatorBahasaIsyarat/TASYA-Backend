# Backend Description

The backend of this project is responsible for capturing video frames from a webcam, processing them using a hand gesture recognition model, and streaming the processed frames to the frontend for display. It is implemented using Python, Flask, OpenCV, TensorFlow, and Mediapipe.

## Functionality
Video Capture: The backend utilizes OpenCV to capture frames from the webcam in real-time. It continuously reads frames from the webcam using the cv2.VideoCapture module.

- Hand Gesture Recognition: Mediapipe's Hands library is used for hand detection and landmark estimation. It processes each frame to detect and extract hand landmarks, which are then used for hand gesture recognition. The backend uses a pre-trained TensorFlow Lite model (tasya.tflite) for predicting the hand gestures based on the extracted landmarks.

- Bounding Box and Labeling: The backend draws a bounding box around the detected hand region in each frame using OpenCV's drawing functions. It also displays the predicted hand gesture label inside the bounding box.

- Video Streaming: The processed frames are converted to JPEG format using OpenCV's cv2.imencode function. These encoded frames are then streamed to the frontend using Flask's Response object with the multipart/x-mixed-replace MIME type. This allows the frontend to continuously receive and display the frames in real-time.

## Optimization

To save network resources, several optimizations have been implemented:

- Frame Resizing: The captured frames are resized to a smaller resolution (e.g., 640x480) before streaming. This reduces the size of each frame and the amount of data transmitted.

- Frame Rate Control: A delay is introduced between frame captures to control the frame rate. The cv2.waitKey function is used to pause the program execution for a specific duration, thus controlling the frame rate.

- Error Handling: Exception handling is implemented to catch any errors that may occur during hand gesture recognition. If an error occurs, such as an empty frame or a frame without a hand detected, it is logged and the processing continues with the next frame.

## Dependencies

The backend relies on the following dependencies, which can be installed via pip:

- Flask: A web framework used for creating the server and handling HTTP requests.
- OpenCV: A computer vision library used for video capture, image processing, and drawing functions.
- TensorFlow: A machine learning framework used for loading and running the hand gesture recognition model.
- Mediapipe: A library for building multimodal machine learning pipelines, utilized here for hand detection and landmark estimation.

## Usage
To run the backend, ensure that you have the required dependencies installed. Then, execute the Python script, and the server will start running. The video stream will be accessible at the specified URL endpoint (/video_feed in this case).

Make sure to update the tasya.tflite model file path if necessary. Additionally, you can adjust the frame resolution, frame rate, and other parameters as needed.

That's a general overview of the backend implementation for this project. Feel free to customize and extend it based on your specific requirements and use case.