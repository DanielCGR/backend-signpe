from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model('model_color.keras')
with open('color_labels.json') as f:
    labels = json.load(f)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    frames = data.get('frames', [])

    if len(frames) < 60:
        return jsonify({'label': 'Not enough frames'}), 400

    sequence = []
    pose_indices = [0, 2, 5] + list(range(7, 17))
    hand_landmarks_count = 21

    for frame_data in frames[:60]:
        header, encoded = frame_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = np.array(image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        pose_results = pose.process(frame_rgb)
        hands_results = hands.process(frame_rgb)

        pose_landmarks = np.zeros((len(pose_indices), 3))
        if pose_results.pose_landmarks:
            pose_landmarks = np.array([
                [pose_results.pose_landmarks.landmark[i].x,
                 pose_results.pose_landmarks.landmark[i].y,
                 pose_results.pose_landmarks.landmark[i].z]
                for i in pose_indices
            ])

        hands_landmarks = np.zeros((2 * hand_landmarks_count, 3))
        if hands_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks[:2]):
                for j, landmark in enumerate(hand_landmarks.landmark):
                    hands_landmarks[i * hand_landmarks_count + j] = [landmark.x, landmark.y, landmark.z]

        frame_data = np.concatenate((pose_landmarks.flatten(), hands_landmarks.flatten()))
        sequence.append(frame_data)

    input_data = np.expand_dims(sequence, axis=0)
    prediction = model.predict(input_data, verbose=0)
    predicted_index = str(np.argmax(prediction))
    predicted_label = labels.get(predicted_index, 'Desconocido')

    return jsonify({'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)