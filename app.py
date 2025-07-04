from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv
from flask_cors import CORS

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

try:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase init failed: {e}")
    raise e

db = firestore.client()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Cargar modelo y labels
MODEL_PATH = 'models-v9/color_v9.keras'
LABELS_PATH = 'models-v9/color_labels.json'

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=True)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)




def add_category_with_signs(custom_category_id, category_name, category_description, category_icon, level_id, signs):
    category_ref = db.collection("categories").document(custom_category_id)
    
    # Step 1: Set or update the category
    category_ref.set({
        "name": category_name,
        "description": category_description,
        "signCount": len(signs),
        "icon": category_icon,
        "levelId" : level_id
    }, merge=True)

    # Step 2: Collect current sign IDs in Firestore for this category
    existing_signs_query = db.collection("signs").where("categoryId", "==", custom_category_id).stream()
    existing_sign_ids = {sign.id for sign in existing_signs_query}

    # Step 3: Collect sign IDs from the input
    new_sign_ids = {sign["id"] for sign in signs if "id" in sign and sign["id"]}

    # Step 4: Delete signs that exist in Firestore but not in the new input
    signs_to_delete = existing_sign_ids - new_sign_ids
    for sign_id in signs_to_delete:
        db.collection("signs").document(sign_id).delete()

    # Step 5: Create or update new/modified signs
    for sign in signs:
        custom_sign_id = sign.get("id")
        sign_name = sign.get("name")
        sign_video_ref = sign.get("videoRef")
        sign_label = sign.get("label")

        if not custom_sign_id or not sign_name:
            continue

        sign_ref = db.collection("signs").document(custom_sign_id)
        sign_ref.set({
            "name": sign_name,
            "label": sign_label,
            "categoryId": custom_category_id,
            "videoRef": sign_video_ref
        }, merge=True)

    return custom_category_id

def add_level(custom_level_id, title, description, difficulty, lessons_count, duration, icon, href):
    level_ref = db.collection("levels").document(custom_level_id)

    level_ref.set({
        "title": title,
        "description": description,
        "difficulty": difficulty,      # This is just a label like "Beginner"
        "lessons": lessons_count,  # This is a number, e.g., 5
        "duration": duration,
        "icon": icon,
        "href": href
    }, merge=True)

    return custom_level_id


def create_all_levels():
    level_id = add_level("levelId01","Nivel Básico","Aprende señas básicas del lenguaje de señas y aprende el alfabeto dactilológico",
                         "Básico",20,"0 horas","🌟","basic")
    print(f"Nivel creado con ID: {level_id}")

    level_id = add_level("levelId02","Nivel Intermedio","Enriquece tu vocabulario con palabras y frases comunes para usar en el día a día",
                         "Intermedio",24,"0 horas","📚","")
    print(f"Nivel creado con ID: {level_id}")

    level_id = add_level("levelId03","Nivel Avanzado","Refuerza tus habilidades para construir oraciones complejas y comunicarte con mayor precisión",
                         "Avanzado",20,"0 horas","💬","")
    print(f"Nivel creado con ID: {level_id}")

    level_id = add_level("levelId04","Nivel Experto","Perfecciona tu conocimiento para interpretar conversaciones a tiempo real y participar en presentaciones formales",
                         "Experto",20,"0 horas","🎓","")
    print(f"Nivel creado con ID: {level_id}")

    level_id = add_level("levelId05","Nivel Experto","Perfecciona tu conocimiento para interpretar conversaciones a tiempo real y participar en presentaciones formales",
                         "Experto",20,"0 horas","🎓","")
    print(f"Nivel creado con ID: {level_id}")

def create_all_categories():
    categoria_id = add_category_with_signs("categoryId01", "Alfabeto", "Aprende el abecedario en LSP y mejora tu habilidad para deletrear con señas.","hand", "levelId01", [
    {"id": "signId001", "name": "Letra A", "label":"a", "videoRef": "YKgCa1dwItA"},
    {"id": "signId002", "name": "Letra B", "label":"b", "videoRef": "nl5ghpTg5ec"},
    {"id": "signId003", "name": "Letra C", "label":"c", "videoRef": "H-anKSubm-w"},
    {"id": "signId004", "name": "Letra D", "label":"d", "videoRef": "r_Gs_Jbdl9E"},
    {"id": "signId005", "name": "Letra E", "label":"e", "videoRef": "youtube.com"},
    {"id": "signId006", "name": "Letra F", "label":"f", "videoRef": "youtube.com"}
    ])
    print(f"Categoría creada con ID: {categoria_id}")
    categoria_id = add_category_with_signs("categoryId02", "Colores", "Identifica y aprende los colores básicos para describir el mundo que te rodea.","palette", "levelId01", [
    {"id": "signId007", "name": "Verde", "label":"verde", "videoRef": "KmUUdxL4W7U"},
    {"id": "signId008", "name": "Rojo", "label":"rojo", "videoRef": "PUx8iIfwvDU"},
    {"id": "signId009", "name": "Amarillo", "label":"amarillo", "videoRef": "y1_EkCMMlhM"},
    {"id": "signId010", "name": "Blanco", "label":"blanco", "videoRef": "1s4aYoAodlc"},
    {"id": "signId011", "name": "Negro", "label":"negro", "videoRef": "60TK3s9V0nY"},
    {"id": "signId012", "name": "Azul", "label":"azul", "videoRef": "VC0csxuR34Q"}
    ])
    print(f"Categoría creada con ID: {categoria_id}")
    categoria_id = add_category_with_signs("categoryId03", "Familia", "Identifica y aprende los colores básicos para describir el mundo que te rodea.","palette", "levelId01", [
    ])
    print(f"Categoría creada con ID: {categoria_id}")
    




#@app.route('/')
#def index():
#    return render_template('index.html')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join("static/dist", path)):
        return send_from_directory("static/dist", path)
    else:
        return send_from_directory("static/dist", "index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    images = data.get('frames')

    if not images or len(images) != 30:
        return jsonify({'error': 'Se requieren exactamente 30 frames'}), 400

    sequence = []

    for img_base64 in images:
        img_bytes = base64.b64decode(img_base64.split(',')[-1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        pose_result = pose.process(image_rgb)
        hands_result = hands.process(image_rgb)

        pose_vec = np.zeros((33, 4))
        if pose_result.pose_landmarks:
            for i, lm in enumerate(pose_result.pose_landmarks.landmark):
                pose_vec[i] = [lm.x, lm.y, lm.z, lm.visibility]

        left_hand = np.full((21, 3), -1.0)
        right_hand = np.full((21, 3), -1.0)

        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for idx, handedness in enumerate(hands_result.multi_handedness):
                label = handedness.classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hands_result.multi_hand_landmarks[idx].landmark])
                if label == 'Left':
                    left_hand = coords
                else:
                    right_hand = coords

        features = np.concatenate([pose_vec.flatten(), left_hand.flatten(), right_hand.flatten()])
        sequence.append(features)

    try:
        input_array = np.expand_dims(np.array(sequence), axis=0)
        prediction = model.predict(input_array, verbose=0)[0]    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error':'Prediction Failed'}),500
    
    
    UID_TEMP = data.get('uid')
    sign_ID = data.get('signId')
    sign_doc = db.collection('signs').document(sign_ID).get()
    current_sign_label = sign_doc.to_dict().get("label") # verde
    category_ID = sign_doc.to_dict().get("categoryId")

    reverse_label_map = {v: int(k) for k, v in labels.items()} #reverse labels to seach index by label
    target_index = reverse_label_map[current_sign_label]    
    probability = float(round(prediction[target_index] * 100, 2))
    
    result = {
        "probability": probability
    }
    
    sign_progress_probability = 0
    signProgress_query = (
    db.collection("signProgress")
    .where("uid", "==", UID_TEMP)
    .where("categoryId","==",category_ID)
    )
    sign_progress_item = next((doc for doc in signProgress_query.stream()if doc.get("signId") == sign_ID), None)
    
    if sign_progress_item is not None and sign_progress_item.exists:
        sign_progress_probability = sign_progress_item.to_dict().get("progress")
        signProgress_doc = db.collection("signProgress").document(sign_progress_item.id)
    else:
        signProgress_doc = db.collection("signProgress").document()
    
    if sign_progress_probability > probability:
        return jsonify(result)
    
    
    signProgress_doc.set({
            "progress": probability,
            "signId": sign_ID,
            "uid": UID_TEMP,
            "categoryId": category_ID
        }, merge=True)
    
    if probability < 0.80 or sign_progress_probability >= 80:#PASAR LUEGO A BASE DE DATOS PARA NO CODIGO DURO
        return jsonify(result)

    
    # CATEGORY PROGRESS 
    category_progress_count = sum(
    1 for doc in signProgress_query.stream()
    if "progress" in doc.to_dict() and doc.to_dict()["progress"] >= 80)

    
    category_doc = db.collection('categories').document(category_ID).get()
    level_ID = category_doc.to_dict().get("levelId")

    category_progress_query = (
    db.collection("categoryProgress")
    .where("levelId", "==", level_ID)
    .where("uid", "==", UID_TEMP)
    ).stream()

    category_progress_doc = next((doc for doc in category_progress_query.stream()if doc.get("categoryId") == category_ID), None)
    #category_progress_doc = next(category_progress_query, None)

    if category_progress_doc:
        doc_ref = db.collection("categoryProgress").document(category_progress_doc.id)
    else:
        doc_ref = db.collection("categoryProgress").document()
    

    doc_ref.set({
    "categoryId": category_ID,
    "levelId": level_ID,
    "progress": category_progress_count,
    "uid": UID_TEMP
    }, merge=True)

    # LEVEL PROGRESS

    total_progress = 0
    for doc in category_progress_query:
        data = doc.to_dict()
        progress = data.get("progress", 0)
        total_progress += progress

    level_progress_ref = (
        db.collection("levelProgress")
        .where("levelId", "==", level_ID)
        .where("uid", "==", UID_TEMP)
        .limit(1)
    ).get()

    if level_progress_ref:
        # Document exists; update it
        doc_snapshot = level_progress_ref[0]
        existing_data = doc_snapshot.to_dict()
        doc_id = doc_snapshot.id

        # Preserve 'available' value if it exists
        updated_data = {
            "progress": total_progress,
            "levelId": level_ID,
            "uid": UID_TEMP,
            "available": existing_data.get("available", False)
        }

        db.collection("levelProgress").document(doc_id).set(updated_data, merge=True)
    else:
        # Create new document if not found
        db.collection("levelProgress").add({
            "progress": total_progress,
            "levelId": level_ID,
            "uid": UID_TEMP,
            "available": True  # default if none exists
        })

    

    return jsonify(result)

@app.route('/get-signs', methods=['GET'])
def get_signs():
    uid = request.args.get('uid')
    category_id = request.args.get('categoryId')

    signProgress_query = (
        db.collection("signProgress")
        .where("uid", "==", uid)
        .where("categoryId", "==", category_id)
    ).stream()

    sign_query = (
        db.collection("signs")
        .where("categoryId", "==", category_id)
    ).stream()

    progress_map = {}
    for doc in signProgress_query:
        data = doc.to_dict()
        sign_id = data.get("signId")
        progress_map[sign_id] = data.get("progress")

    # Step 2: Merge sign data with progress (if available)
    merged_results = []
    for doc in sign_query:
        sign_data = doc.to_dict()
        sign_id = doc.id
        
        # Inject progress if found
        sign_data["progress"] = progress_map.get(sign_id, 0)
        merged_results.append({**sign_data, "id": sign_id})


    return jsonify(merged_results)

@app.route('/get-categories', methods=['GET'])
def get_categories():
    uid = request.args.get('uid')
    level_id = request.args.get('levelId')

    categoryProgress_query = (
        db.collection("categoryProgress")
        .where("uid", "==", uid)
    ).stream()

    category_query = (
        db.collection("categories")
        .where("levelId", "==", level_id)
    ).stream()
    
    progress_map = {}
    for doc in categoryProgress_query:
        data = doc.to_dict()
        category_id = data.get("categoryId")
        progress_map[category_id] = data.get("progress")

    merged_results = []
    for doc in category_query:
        category_data = doc.to_dict()
        category_id = doc.id
        
        # Inject progress if found
        category_data["progress"] = progress_map.get(category_id, 0)
        merged_results.append({**category_data, "id": category_id})

    return jsonify(merged_results)

@app.route('/get-levels', methods=['GET'])
def get_levels():
    uid = request.args.get('uid')
    
    levelProgress_query = (
        db.collection("levelProgress")
        .where("uid", "==", uid)
    ).stream()

    level_query = (
        db.collection("levels")
    ).stream()

    progress_map = {}
    for doc in levelProgress_query:
        data = doc.to_dict()
        level_id = data.get("levelId")
        progress_map[level_id] = {
            "progress": data.get("progress", 0),
            "available": data.get("available", False)
        }

    merged_results = []
    for doc in level_query:
        level_data = doc.to_dict()
        level_id = doc.id
        
        # Inject progress if found
        progress = progress_map.get(level_id, {}).get("progress", 0)
        available = progress_map.get(level_id, {}).get("available", False)
        merged_results.append({
            **level_data,
            "id": level_id,
            "progress": progress,
            "available": available
        })

    return jsonify(merged_results)


@app.route('/get-exam-signs', methods=['GET'])
def get_exam_signs():
    level_id = "levelId01"#request.args.get('levelId')

    category_query = (
        db.collection("categories")
        .where("levelId", "==", level_id)
    ).stream()

    category_ids = [category_doc.id for category_doc in category_query]
    merged_results = []

    for category_id in category_ids:
        signs_query = (
            db.collection("signs")
            .where("categoryId", "==", category_id)
        ).stream()

        for sign_doc in signs_query:
            merged_results.append(sign_doc.to_dict())

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(merged_results)

    return jsonify(merged_results)


if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 10000))
    #app.run(host='0.0.0.0', port=port)
    #app.run(debug=True, host='127.0.0.1', port=5000)
    
    #create_all_categories()

    #get_sign_progress()
    #get_categories()

    #create_all_levels()
    #get_levels()
    get_exam_signs()

[{'category': {'description': 'Aprende el abecedario en LSP y mejora tu habilidad para deletrear con señas.', 'icon': 'hand', 'signCount': 6, 'levelId': 'levelId01', 'name': 'Alfabeto'}, 'signs': [{'videoRef': 'YKgCa1dwItA', 'label': 'a', 'categoryId': 'categoryId01', 'name': 'Letra A'}, {'videoRef': 'nl5ghpTg5ec', 'label': 'b', 'categoryId': 'categoryId01', 'name': 'Letra B'}, {'videoRef': 'H-anKSubm-w', 'label': 'c', 'categoryId': 'categoryId01', 'name': 'Letra C'}, {'videoRef': 'r_Gs_Jbdl9E', 'label': 'd', 'categoryId': 'categoryId01', 'name': 'Letra D'}, {'videoRef': 'youtube.com', 'label': 'e', 'categoryId': 'categoryId01', 'name': 
'Letra E'}, {'videoRef': 'youtube.com', 'label': 'f', 'categoryId': 'categoryId01', 'name': 'Letra F'}]}, 
{'category': {'description': 'Identifica y aprende los colores básicos para describir el mundo que te rodea.', 'icon': 'palette', 'signCount': 6, 'levelId': 'levelId01', 'name': 'Colores'}, 'signs': [{'videoRef': 'KmUUdxL4W7U', 'label': 'verde', 'categoryId': 'categoryId02', 'name': 'Verde'}, {'videoRef': 'PUx8iIfwvDU', 'label': 'rojo', 'categoryId': 'categoryId02', 'name': 'Rojo'}, {'videoRef': 'y1_EkCMMlhM', 'label': 'amarillo', 'categoryId': 'categoryId02', 'name': 'Amarillo'}, {'videoRef': '1s4aYoAodlc', 'label': 'blanco', 'categoryId': 'categoryId02', 'name': 'Blanco'}, {'videoRef': '60TK3s9V0nY', 'label': 'negro', 'categoryId': 'categoryId02', 'name': 'Negro'}, {'videoRef': 'VC0csxuR34Q', 'label': 'azul', 'categoryId': 'categoryId02', 'name': 'Azul'}]}]

[{'videoRef': 'YKgCa1dwItA', 'label': 'a', 'categoryId': 'categoryId01', 'name': 'Letra A'},{'videoRef': 'YKgCa1dwItA', 'label': 'a', 'categoryId': 'categoryId01', 'name': 'Letra A'},{'videoRef': 'YKgCa1dwItA', 'label': 'a', 'categoryId': 'categoryId01', 'name': 'Letra A'}]