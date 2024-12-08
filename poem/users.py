from flask import request, jsonify, Flask
from bson import ObjectId

from server import app, cipher_suite
from test import user_follow, get_user_info_by_uid, user_unfollow, get_uid


# ruta, recibe el id de un usuario y lo sigue
@app.route('/follow', methods=['POST'])
def follow_user():
    token = request.headers.get('Authorization')

    if not token:
        return jsonify({'error': 'Missing token'}), 401

    decrypted_token = cipher_suite.decrypt(token.encode()).decode()
    current_user_id = decrypted_token.split(':')[1]

    user_id = request.args.get('user_id')

    if not user_follow(current_user_id, user_id):
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'message': 'User followed successfully'}), 200


# ruta, recibe el id de un usuario y lo deja de seguir
@app.route('/unfollow', methods=['POST'])
def unfollow_user():
    token = request.headers.get('Authorization')

    if not token:
        return jsonify({'error': 'Missing token'}), 401

    decrypted_token = cipher_suite.decrypt(token.encode()).decode()
    current_user_id = decrypted_token.split(':')[1]

    user_id = request.args.get('user_id')

    if not user_unfollow(current_user_id, user_id):
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'message': 'User unfollowed successfully'}), 200


# ruta, recibe el id de un usuario y retorna la informacion de ese usuario
@app.route('/getUserInfo', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    user_info = get_user_info_by_uid(user_id)

    if not user_info:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user': user_info})


# ruta, recibe el correo de un usuario y retorna el id de ese usuario
@app.route('/getUserId', methods=['GET'])
def get_user_id():
    email = request.args.get('correo')
    if not email:
        return jsonify({'error': 'Correo is required'}), 400

    user = get_uid(email)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user_id': str(user['_id'])})

