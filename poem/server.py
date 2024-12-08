from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
from cryptography.fernet import Fernet
from datetime import datetime

from test import get_user_by_email_password, set_user

app = Flask(__name__)


# Cargar o generar la clave secreta
try:
    with open('secret.key', 'rb') as f:
        SECRET_KEY = f.read()
except FileNotFoundError:
    SECRET_KEY = Fernet.generate_key()
    with open('secret.key', 'wb') as f:
        f.write(SECRET_KEY)

cipher_suite = Fernet(SECRET_KEY)



# Ruta para crear una cuenta de usuario
@app.route('/createAccount', methods=['POST'])
def create_user():
    data = request.get_json()
    print('Received:', data)
    username = data.get('usuario')
    email = data.get('correo')
    password = data.get('contra')
    foto_perfil_url = data.get('foto_perfil_url')  # Recibir la URL de la foto de perfil

    if not username or not email or not password:
        return jsonify({'error': 'Missing fields'}), 400

    user_id = set_user(username, email, password, foto_perfil_url, datetime.now().strftime('%d/%m/%y - %H:%M'))

    if user_id == -1:
        return jsonify({'error': 'User already exists'}), 409

    return jsonify({'message': 'User created successfully'}), 201


# Ruta para iniciar sesión
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('correo')
    password = data.get('contra')

    if not email or not password:
        return jsonify({'error': 'Missing fields'}), 400

    user = get_user_by_email_password(email, password)

    if not user:
        print('Invalid credentials')
        return jsonify({'error': 'Invalid credentials'}), 401

    token = f"user_id:{user['_id']}"
    encrypted_token = cipher_suite.encrypt(token.encode()).decode()

    return jsonify({'message': 'Login successful', 'token': encrypted_token}), 200



from users import *
from poems import *

# Iniciar el servidor
if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5001)
