from flask import Flask, request, jsonify, send_file
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Cargar variables de entorno y configurar LangChain
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La clave de API no está configurada en las variables de entorno.")
llm = OpenAI(temperature=0.7, openai_api_key=api_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Ruta para servir el archivo HTML del chat
@app.route("/", methods=["GET"])
def home():
    return send_file("index.html")

# Ruta para manejar mensajes del chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Si es el primer mensaje, saludar (Asi se comprueba que el chat este conectado al servidor.)
    if user_input.lower() == "init":
        return jsonify({"response": "Hola, ¿en qué te puedo ayudar?"})

    try:
        response = conversation.run(input=user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
