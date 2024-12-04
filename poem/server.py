from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)


# Configuración de la base de datos MongoDB
def connect_db():
    client = MongoClient('mongodb://192.168.0.207:27017/')  # Cambia según tu configuración
    db = client['poems']  # Nombre de la base de datos
    return db


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

    db = connect_db()
    users_collection = db['users']

    if users_collection.find_one({'correo': email}):
        return jsonify({'error': 'Email already exists'}), 409  # Código 409: Conflicto

    new_user = {
        'user': username,
        'correo': email,
        'contra': password,
        'foto_perfil_url': foto_perfil_url,
        'fecha_registro': datetime.now().isoformat()
    }
    users_collection.insert_one(new_user)

    return jsonify({'message': 'User created successfully'}), 201


# Ruta para iniciar sesión
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print('Received:', data)
    email = data.get('correo')
    password = data.get('contra')

    if not email or not password:
        return jsonify({'error': 'Missing fields'}), 400

    db = connect_db()
    users_collection = db['users']
    user = users_collection.find_one({'correo': email, 'contra': password})

    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Login successful'}), 200


# Ruta para obtener todos los poemas de todos los usuarios
@app.route('/getPoems', methods=['GET'])
def get_poems():
    db = connect_db()
    posts_collection = db['posts']
    users_collection = db['users']

    # Consultar los poemas y añadir datos de usuario
    poems = posts_collection.find().sort('fecha_publicacion', -1)
    poem_list = []

    for poem in poems:
        user = users_collection.find_one({'_id': poem['user_id']})
        poem_list.append({
            'id': str(poem['_id']),
            'user_id': str(poem['user_id']),
            'titulo': poem['titulo'],
            'poema': poem['poema'],
            'descripcion': poem['descripcion'],
            'fecha_publicacion': poem['fecha_publicacion'],
            'foto_perfil_url': user['foto_perfil_url'] if user else None,
            'usuario': user['user'] if user else 'Unknown'
        })

    if not poem_list:
        return jsonify({'message': 'No poems found'}), 404

    return jsonify({'poems': poem_list})


# Ruta para obtener información de un usuario, incluida la foto de perfil
@app.route('/getUserInfo', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    db = connect_db()
    users_collection = db['users']
    user = users_collection.find_one({'_id': ObjectId(user_id)})

    if not user:
        return jsonify({'error': 'User not found'}), 404

    user_info = {
        'id': str(user['_id']),
        'user': user['user'],
        'correo': user['correo'],
        'foto_perfil_url': user['foto_perfil_url'],
        'fecha_registro': user['fecha_registro']
    }

    return jsonify({'user': user_info})


# Ruta para crear un poema
@app.route('/createPoem', methods=['POST'])
def create_poem():
    data = request.get_json()
    print('Received:', data)

    user_id = data.get('user_id')  # ID del usuario que está creando el poema
    titulo = data.get('titulo')  # Título del poema
    poema = data.get('poema')  # Contenido del poema
    descripcion = data.get('descripcion')  # Descripción del poema

    if not user_id or not titulo or not poema or not descripcion:
        return jsonify({'error': 'Missing fields'}), 400

    db = connect_db()
    posts_collection = db['posts']
    new_post = {
        'user_id': ObjectId(user_id),
        'titulo': titulo,
        'poema': poema,
        'descripcion': descripcion,
        'fecha_publicacion': datetime.now().strftime('%d/%m/%y %H:%M')
    }
    posts_collection.insert_one(new_post)

    return jsonify({'message': 'Poem created successfully'}), 201

# ruta, recibe el correo de un usuario y retorna el id de ese usuario
@app.route('/getUserId', methods=['GET'])
def get_user_id():
    email = request.args.get('correo')
    if not email:
        return jsonify({'error': 'Correo is required'}), 400

    db = connect_db()
    users_collection = db['users']
    user = users_collection.find_one({'correo': email})

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user_id': str(user['_id'])})




# Iniciar el servidor
if __name__ == '__main__':
    from datetime import datetime
    app.run(debug=True, host='0.0.0.0', port=5001)
