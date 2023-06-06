import cv2
from flask import Flask, render_template, Response
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading

app = Flask(__name__)

cap = cv2.VideoCapture(0)

frame_width = 640  # Set the desired frame width
frame_height = 480  # Set the desired frame height

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

frame_delay = 100  # Set the desired frame delay (in milliseconds)

frame_lock = threading.Lock()

interpreter = tf.lite.Interpreter(model_path='tasya.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

def generate_frames():
    while True:
        success, frame = cap.read()

        if not success or frame is None:
            continue

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

                if roi.size == 0:
                    continue

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

                label_text = f"{pred_label}"
                cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error: {str(e)}")
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
