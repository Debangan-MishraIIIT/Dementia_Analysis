from flask import Flask, request, jsonify
from model_handlers import get_model, get_predictions
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
model= None

def load_model():
    global model
    if model is None:
        model = get_model()  # Function to load your machine learning model

@app.route("/")
def health_check():
    return "Health Check Succesful"

@app.route("/image", methods=['POST'])
def get_output():
    load_model()

    try:
        uploaded_file = request.files['file']
    except:
        return jsonify({"error": "File Read Error"}), 400

    if uploaded_file.filename == '':
        return jsonify({"error": "No File Uploaded"}), 400
    elif (uploaded_file.filename).split(".")[-1] not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid File Extension"}), 400
    else:
        img_bytes= uploaded_file.read()
        img= Image.open(BytesIO(img_bytes)).convert("L")
        try:
            prediction= get_predictions(img, model)
            return jsonify({"message": f"{prediction}"}), 200
        except:
            return jsonify({"error": "Model Error"}), 400

if __name__=="__main__":
    app.run(debug=True)