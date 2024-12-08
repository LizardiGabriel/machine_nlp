from datetime import datetime
from pymongo import MongoClient

def connect_db():
    client = MongoClient('mongodb://127.0.0.1:27017/')
    db = client['poems']
    return client, db

def get_user_by_email_password(email, password):
    client, db = connect_db()
    try:
        users_collection = db['users']
        user = users_collection.find_one({'correo': email, 'contra': password})
        return user
    finally:
        client.close()

def drop_db():
    client, db = connect_db()
    try:
        db.drop_collection('users')
        db.drop_collection('posts')
    finally:
        client.close()

# Users

def exists(user_id):
    client, db = connect_db()
    try:
        users_collection = db['users']
        user = users_collection.find_one({'_id': user_id})
        return user is not None
    finally:
        client.close()

def get_users():
    client, db = connect_db()
    try:
        users_collection = db['users']
        users = list(users_collection.find())
        return users
    finally:
        client.close()

def set_user(username, email, password, foto_perfil_url, date):
    client, db = connect_db()
    try:
        users_collection = db['users']
        if users_collection.find_one({'correo': email}):
            return -1
        new_user = {
            'user': username,
            'correo': email,
            'contra': password,
            'foto_perfil_url': foto_perfil_url,
            'fecha_registro': date
        }
        users_collection.insert_one(new_user)
        return new_user['_id']
    finally:
        client.close()

def get_uid(email):
    client, db = connect_db()
    try:
        users_collection = db['users']
        user = users_collection.find_one({'correo': email})
        if not user:
            return None
        return user['_id']
    finally:
        client.close()

def user_follow(user_id, user_id_to_follow):
    if not exists(user_id) or not exists(user_id_to_follow):
        return False

    client, db = connect_db()
    try:
        users_collection = db['users']
        users_collection.update_one(
            {'_id': user_id},
            {'$addToSet': {'following': user_id_to_follow}}
        )
        return True
    finally:
        client.close()

def user_unfollow(user_id, user_id_to_unfollow):
    if not exists(user_id) or not exists(user_id_to_unfollow):
        return False

    client, db = connect_db()
    try:
        users_collection = db['users']
        users_collection.update_one(
            {'_id': user_id},
            {'$pull': {'following': user_id_to_unfollow}}
        )
        return True
    finally:
        client.close()

def get_user_info_by_uid(user_id):
    if not exists(user_id):
        return None

    client, db = connect_db()
    try:
        users_collection = db['users']
        user = users_collection.find_one({'_id': user_id})
        user_info = {
            'id': str(user['_id']),
            'user': user['user'],
            'correo': user['correo'],
            'foto_perfil_url': user['foto_perfil_url'],
            'fecha_registro': user['fecha_registro'],
            'following': user.get('following', [])
        }
        return user_info
    finally:
        client.close()

# Poems

def get_poems():
    client, db = connect_db()
    try:
        posts_collection = db['posts']
        poems = list(posts_collection.find())
        return poems
    finally:
        client.close()

def set_poem(user_id, titulo, poema, descripcion, date):
    if not exists(user_id):
        return False

    client, db = connect_db()
    try:
        posts_collection = db['posts']
        new_post = {
            'user_id': user_id,
            'titulo': titulo,
            'poema': poema,
            'descripcion': descripcion,
            'fecha_publicacion': date,
            'likes': [],
            'num_comentarios': 0
        }
        posts_collection.insert_one(new_post)
        return new_post['_id']
    finally:
        client.close()

def get_poems_following(user_id):
    if not exists(user_id):
        return None

    client, db = connect_db()
    try:
        users_collection = db['users']
        posts_collection = db['posts']
        user = users_collection.find_one({'_id': user_id})
        following = user.get('following', [])
        poems = posts_collection.find({'user_id': {'$in': following}}).sort('fecha_publicacion', -1)
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
                'usuario': user['user'] if user else 'Unknown',
                'likes': poem.get('likes', []),
                'num_comentarios': poem.get('num_comentarios', 0)
            })
        return poem_list
    finally:
        client.close()
