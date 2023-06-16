# TASYA-API Documentation

TASYA-API is a Flask-based backend that provides real-time video processing and
gesture recognition using TensorFlow Lite and MediaPipe. It receives video
frames from a client application, performs hand gesture recognition, and sends
back the processed frames with gesture labels.

## Installation

To use TASYA-API, you need to set up the required dependencies and run the
backend server.

### Prerequisites

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- Flask
- Flask-SocketIO
- OpenCV (cv2)
- NumPy
- MediaPipe

You can install the required packages using pip:

```pip
pip install tensorflow flask flask-socketio opencv-python numpy mediapipe
```

### Setting Up

1. Copy the provided code into a file named app.py.
2. Download the TensorFlow Lite model (tasya.tflite) and place it in the same
   directory as app.py.
3. Run the backend server using the following command:

```terminal
python app.py
```

By default, the server will listen on `http://localhost:5000`.

## API Endpoints

/

- Method: GET
- Description: Returns a simple message to verify that the API is running.
- Example: `http://localhost:5000/`

### WebSocket Events

connect

- Event: 'connect'
- Description: Triggered when a client connects to the server using WebSocket.
- Example:

```js
socket.on("connect", function () {
  console.log("Connected to the server");
});
```

disconnect

- Event: 'disconnect'
- Description: Triggered when a client disconnects from the server using
  WebSocket.
- Example:

```js
socket.on("disconnect", function () {
  console.log("Disconnected from the server");
});
```

video_stream

- Event: 'video_stream'
- Description: Receives video frames from the client application for processing.
- Data:
  - frame: Base64-encoded string of the video frame.
  - width: Width of the video frame.
  - height: Height of the video frame.
- Example:

```js
// Send video frame data to the server
socket.emit("video_stream", {
  frame: "base64-encoded-frame-data",
  width: 640,
  height: 480,
});
```

## WebSocket Responses

processed_frame

- Event: 'processed_frame'
- Description: Sends back the processed video frame with gesture labels.
- Data:
  - frame: Base64-encoded string of the processed video frame.
- Example:

```js
// Receive processed video frame data from the server
socket.on("processed_frame", function (data) {
  var frameData = data.frame;
  // Process the frame data as needed
});
```

## Processing Video Frames

The process_frame function is responsible for processing each video frame
received from the client. It performs the following steps:

1. Resize the frame to the desired dimensions (frame_width and frame_height).
2. Perform hand detection and landmark estimation using MediaPipe.
3. Extract the bounding box coordinates around the detected hand.
4. Crop the frame using the bounding box coordinates.
5. Preprocess the cropped region for gesture recognition.
6. Pass the preprocessed data through the TensorFlow Lite model.
7. Retrieve the predicted gesture label.
8. Draw the bounding box and gesture label on the frame.
9. Return the processed frame.
10. Feel free to modify the process_frame function to adapt it to your specific
    requirements.

## Usage Example

Here's an example of how you can use TASYA-API in a client application:

```python
import cv2
import base64
import requests
import json

# Read a video file or capture frames from a camera
cap = cv2.VideoCapture(0)

# Define the server URL
server_url = 'http://localhost:5000/video_stream'

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Encode the frame as base64
    _, frame_data = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(frame_data).decode('utf-8')

    # Prepare the payload
    payload = {
        'frame': frame_base64,
        'width': frame.shape[1],
        'height': frame.shape[0]
    }

    # Send the video frame to the server
    response = requests.post(server_url, json=payload)
    if response.status_code == 200:
        # Receive the processed frame from the server
        data = response.json()
        processed_frame_base64 = data['frame']

        # Decode the processed frame
        processed_frame_data = base64.b64decode(processed_frame_base64)
        processed_frame = cv2.imdecode(np.frombuffer(processed_frame_data, np.uint8), cv2.IMREAD_COLOR)

        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
```

This example demonstrates how to send video frames to the TASYA-API server and
display the processed frames received from the server.

## Conclusion

This concludes the documentation for TASYA-API. You can integrate it into your
applications to perform real-time gesture recognition using video streams. Feel
free to customize and extend the code to suit your specific needs.
