from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from main import generate_ai_response

app = Flask(__name__)
CORS(app)  

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])

def chat():
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"reply": "Please enter a message."}), 400
        
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"reply": "Please enter a valid message."}), 400
    
    try:
        bot_reply = generate_ai_response(user_message)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"reply": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)