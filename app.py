import base64
import threading
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

frame_lock = threading.Lock()

interpreter = tf.lite.Interpreter(model_path='tasya.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

label_mapping = {
    29: 'Other',  # Default label for folders other than 'A' to 'Z', 'del', 'nothing', 'space'
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'del', 27: 'nothing', 28: 'space'
}

def process_frame(frame, frame_width, frame_height):
    if frame is None or frame.size == 0:
        return frame

    frame = cv2.resize(frame, (frame_width, frame_height))

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    try:
        if hand_landmarks:
            handLMs = hand_landmarks[0]  # Get the first hand landmark only

            h, w, c = frame.shape

            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (64, 64))
                roi_normalized = roi_resized / 255.0
                roi_final = np.expand_dims(roi_normalized, axis=-1)
                input_data = np.expand_dims(roi_final, axis=0).astype(np.float32)

                with frame_lock:
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])

                pred_index = np.argmax(output_data)
                pred_label = label_mapping.get(pred_index, "Unknown")

                label_text = f"Character: {pred_label}"
                cv2.putText(frame, label_text, (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {str(e)}")

    return frame


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_stream')
def handle_video_stream(data):
    frame_bytes = data['frame']
    frame_width = data['width']  # Get the width from the request
    frame_height = data['height']  # Get the height from the request
    frame_data = np.frombuffer(base64.b64decode(frame_bytes), dtype=np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    
    # Use the frame_width and frame_height variables as needed in your code

    processed_frame = process_frame(frame, frame_width, frame_height)

    if processed_frame is not None and processed_frame.size > 0:
        ret, buffer = cv2.imencode('.jpg', processed_frame)

        if ret:
            frame_bytes = buffer.tobytes()
            encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')

            emit('processed_frame', {'frame': encoded_frame})
    else:
        emit('processed_frame', {'frame': ''})

    # Notify when someone hits the video_stream endpoint
    print(f'endpoint hit: {data}')

    # You can add your notification logic or trigger specific tasks here

@app.route('/')
def index():
    return "TASYA-API is running!"

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
