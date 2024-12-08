from flask import request, jsonify
from bson import ObjectId
from server import app, cipher_suite
from datetime import datetime

from test import *


# Ruta para crear un poema
@app.route('/createPoem', methods=['POST'])
def create_poem():
    data = request.get_json()
    print('Received:', data)

    user_id = data.get('user_id')
    titulo = data.get('titulo')
    poema = data.get('poema')
    descripcion = data.get('descripcion')

    if not user_id or not titulo or not poema or not descripcion:
        return jsonify({'error': 'Missing fields'}), 400

    if not set_poem(user_id, titulo, poema, descripcion, datetime.now().strftime('%d/%m/%y - %H:%M')):
        return jsonify({'error': 'Invalid user ID'}), 401

    return jsonify({'message': 'Poem created successfully'}), 201


# Ruta para obtener todos los poemas de todos los usuarios
@app.route('/getPoems', methods=['GET'])
def get_poems():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is required'}), 401

    print('token: ', token)

    decrypted_token = cipher_suite.decrypt(token.encode()).decode()
    user_id = decrypted_token.split(':')[1]

    print('user_id: ', user_id)

    poem_list = get_poems_following(user_id)
    if not poem_list:
        print('No poems found')
        return jsonify({'message': 'No poems found'}), 404

    return jsonify({'poems': poem_list})