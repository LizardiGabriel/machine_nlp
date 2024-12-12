from datetime import datetime

from bson import ObjectId
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
    user_id = ObjectId(user_id)
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
    user_id = ObjectId(user_id)
    user_id_to_follow = ObjectId(user_id_to_follow)
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
    user_id = ObjectId(user_id)
    user_id_to_unfollow = ObjectId(user_id_to_unfollow)

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
    user_id = ObjectId(user_id)
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
    user_id = ObjectId(user_id)
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
            'likes': 0,
            'num_comentarios': 0
        }
        posts_collection.insert_one(new_post)
        return new_post['_id']
    finally:
        client.close()



def get_poems_by_followed_users(user_id):
    user_id = ObjectId(user_id)
    if not exists(user_id):
        return None

    client, db = connect_db()
    try:
        users_collection = db['users']
        posts_collection = db['posts']

        user = users_collection.find_one({'_id': user_id})
        followed_users = user.get('following', [])

        poems = list(posts_collection.find({'user_id': {'$in': followed_users}}))

        for poema in poems:
            user_info = users_collection.find_one({'_id': poema['user_id']})
            poema['user'] = user_info['user']
            poema['foto_perfil_url'] = user_info['foto_perfil_url']

        # retornar un json serializable
        poems = [
            {
                'id': str(poema['_id']),
                'user_id': str(poema['user_id']),
                'titulo': poema['titulo'],
                'poema': poema['poema'],
                'descripcion': poema['descripcion'],
                'fecha_publicacion': poema['fecha_publicacion'],
                'foto_perfil_url': poema['foto_perfil_url'],
                'usuario': poema['user'],
                'num_likes': poema['likes'],
                'num_comentarios': poema['num_comentarios']
            }
            for poema in poems
        ]

        print(poems)

        return poems
    finally:
        client.close()




