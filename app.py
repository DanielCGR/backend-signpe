from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from io import BytesIO
from PIL import Image
import base64
import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

firebase_config = {
    "type": os.environ.get("FIREBASE_TYPE"),
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": os.environ.get("FIREBASE_AUTH_URI"),
    "token_uri": os.environ.get("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.environ.get("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": os.environ.get("FIREBASE_UNIVERSE_DOMAIN")
}

cred = credentials.Certificate(firebase_config)

firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

# Load model and labels
#model = tf.keras.models.load_model('model_color.keras')
with open('color_labels.json') as f:
    labels = json.load(f)

# Initialize MediaPipe
#mp_pose = mp.solutions.pose
#mp_hands = mp.solutions.hands
#pose = mp_pose.Pose()
#hands = mp_hands.Hands()

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

    #input_data = np.expand_dims(sequence, axis=0)
    #prediction = model.predict(input_data, verbose=0)
    input_data = np.expand_dims(sequence, axis=0)
    prediction = model.predict(input_data, verbose=0)[0]

    #predicted_index = str(np.argmax(prediction))
    #predicted_label = labels.get(predicted_index, 'Desconocido')
    predicted_index = int(np.argmax(prediction))
    predicted_label = labels.get(str(predicted_index), 'Desconocido')
    confidence = {labels[str(i)]: float(round(prediction[i] * 100, 2)) for i in range(len(prediction))}

    db.collection('predictions').add({
    'label': predicted_label,
    'confidence': confidence,
    'timestamp': firestore.SERVER_TIMESTAMP
    })

    #return jsonify({'label': predicted_label})
    return jsonify({
        'label': predicted_label,
        'confidence': confidence
    })

# Create
@app.route('/add_prediction', methods=['POST'])
def add_prediction():
    data = request.json
    doc_ref = db.collection('predictions').add(data)
    return jsonify({'id': doc_ref[1].id, 'message': 'Prediction added'}), 201

# Read All
@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    docs = db.collection('predictions').stream()
    result = [{doc.id: doc.to_dict()} for doc in docs]
    return jsonify(result)

# Read One
@app.route('/get_prediction/<id>', methods=['GET'])
def get_prediction(id):
    doc = db.collection('predictions').document(id).get()
    if doc.exists:
        return jsonify(doc.to_dict())
    else:
        return jsonify({'error': 'Not found'}), 404

# Update
@app.route('/update_prediction/<id>', methods=['PUT'])
def update_prediction(id):
    data = request.json
    db.collection('predictions').document(id).update(data)
    return jsonify({'message': 'Prediction updated'})

# Delete
@app.route('/delete_prediction/<id>', methods=['DELETE'])
def delete_prediction(id):
    db.collection('predictions').document(id).delete()
    return jsonify({'message': 'Prediction deleted'})


@app.route('/get_users', methods=['GET'])
def get_users():
    users_ref = db.collection('users')
    docs = users_ref.stream()

    users = []
    for doc in docs:
        user_data = doc.to_dict()
        user_data['id'] = doc.id  # Include document ID if needed
        users.append(user_data)

    return jsonify(users)

if __name__ == '__main__':
    app.run(debug=True)