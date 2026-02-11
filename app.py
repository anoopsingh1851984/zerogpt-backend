 from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Hugging Face Token Render Environment में रहेगा
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    return jsonify(response.json())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
