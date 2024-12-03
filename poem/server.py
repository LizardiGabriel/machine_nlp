# Instalar el conector de MySQL
# pip install mysql-connector-python

from flask import Flask, request, jsonify
import mysql.connector

from mysql.connector import IntegrityError  # Importar excepción específica

app = Flask(__name__)

# Conectar a la base de datos MySQL
def connect_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='poems'
    )

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

    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO Users (user, correo, contra, foto_perfil_url) VALUES (%s, %s, %s, %s)',
                           (username, email, password, foto_perfil_url))
            conn.commit()
    except IntegrityError as e:
        # Manejar error de entrada duplicada
        if 'Duplicate entry' in str(e):
            return jsonify({'error': 'Email already exists'}), 409  # Código 409: Conflicto
        else:
            return jsonify({'error': 'Database error', 'details': str(e)}), 500

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

    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Users WHERE correo = %s AND contra = %s', (email, password))
        user = cursor.fetchone()

    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Login successful'}), 200


# Ruta para obtener todos los poemas de todos los usuarios
@app.route('/getPoems', methods=['GET'])
def get_poems():
    with connect_db() as conn:
        cursor = conn.cursor()
        # Consulta con JOIN para obtener la foto del perfil del usuario
        cursor.execute('''
            SELECT 
                Posts.id, 
                Posts.user_id, 
                Posts.titulo, 
                Posts.poema, 
                Posts.descripcion, 
                Posts.fecha_publicacion,
                Users.foto_perfil_url,
                Users.user
            FROM Posts
            INNER JOIN Users ON Posts.user_id = Users.id
            ORDER BY Posts.fecha_publicacion DESC
        ''')
        poems = cursor.fetchall()

    if not poems:
        return jsonify({'message': 'No poems found'}), 404

    # Convertir los resultados a una lista de diccionarios
    poem_list = []
    for poem in poems:
        poem_list.append({
            'id': poem[0],
            'user_id': poem[1],  # ID del usuario
            'titulo': poem[2],
            'poema': poem[3],
            'descripcion': poem[4],
            'fecha_publicacion': poem[5].strftime('%Y-%m-%d %H:%M:%S'),
            'foto_perfil_url': poem[6],
            'usuario': poem[7]
        })

    return jsonify({'poems': poem_list})




# Ruta para obtener información de un usuario, incluida la foto de perfil
@app.route('/getUserInfo', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Users WHERE id = %s', (user_id,))
        user = cursor.fetchone()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    user_info = {
        'id': user[0],
        'user': user[1],
        'correo': user[2],
        'foto_perfil_url': user[4],  # URL de la foto de perfil
        'fecha_registro': user[5].strftime('%Y-%m-%d %H:%M:%S')
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

    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO Posts (user_id, titulo, poema, descripcion, fecha_publicacion) VALUES (%s, %s, %s, %s, NOW())',
                       (user_id, titulo, poema, descripcion))
        conn.commit()

    return jsonify({'message': 'Poem created successfully'}), 201




# Iniciar el servidor
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
