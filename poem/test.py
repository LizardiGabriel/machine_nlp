import json
import requests

# URL base del servidor
BASE_URL = 'http://localhost:5001'

# Función para crear un usuario
def create_user(username, email, password, foto_perfil_url):
    url = f'{BASE_URL}/createAccount'
    data = {
        'usuario': username,
        'correo': email,
        'contra': password,
        'foto_perfil_url': foto_perfil_url
    }

    response = requests.post(url, json=data)
    if response.status_code == 201:
        print('User created successfully')
        return response.json()  # Retorna la respuesta con los detalles del usuario
    else:
        print('Error creating user:', response.json())
        return None

# Función para crear un poema
def create_poem(user_id, titulo, poema, descripcion):
    url = f'{BASE_URL}/createPoem'
    data = {
        'user_id': user_id,
        'titulo': titulo,
        'poema': poema,
        'descripcion': descripcion
    }

    response = requests.post(url, json=data)
    if response.status_code == 201:
        print('Poem created successfully')
    else:
        print('Error creating poem:', response.json())

# Función para obtener todos los poemas
def get_poems():
    url = f'{BASE_URL}/getPoems'
    response = requests.get(url)
    if response.status_code == 200:
        poems = response.json().get('poems', [])
        print(f'Found {len(poems)} poems:')
        for poem in poems:
            print(f"Title: {poem['titulo']}, Author: {poem['usuario']}, Date: {poem['fecha_publicacion']}")
    else:
        print('Error fetching poems:', response.json())

# Función para obtener la información de un usuario
def get_user_info(user_id):
    url = f'{BASE_URL}/getUserInfo?user_id={user_id}'
    response = requests.get(url)
    if response.status_code == 200:
        user_info = response.json().get('user', {})
        print(f"User Info: {user_info}")
    else:
        print('Error fetching user info:', response.json())


def get_uid(email):
    url = f'{BASE_URL}/getUserId?correo={email}'
    response = requests.get(url)
    if response.status_code == 200:
        user_id = response.json().get('user_id')
        return user_id
    else:
        print('Error fetching user ID:', response.json())
        return None



if __name__ == '__main__':

    poema1 = 'Nunca conocí la felicidad;\nNo creía que los sueños se hicieran realidad;\nNo podía creer realmente que estaba enamorado;\nHasta que finalmente te conocí'
    poema2 = 'Amarte no tiene fin ni principio;\nAmarte lo es todo;\nEs infinito en el tiempo;\nEs lo que siento por ti'
    poema3 = 'El amor es un sentimiento;\nQue no se puede explicar;\nEs un sentimiento que se siente;\nY que se debe de respetar'
    poema4 = 'Te amo con todo mi corazón;\nEres mi razón de ser;\nEres mi razón de vivir;\nEres mi razón de amar'
    poema5 = 'tus ojos son como el mar;\nque me hacen soñar;\nme hacen volar;\nme hacen amar'

    foto_perfil1 = 'https://i.pinimg.com/736x/b1/f4/c4/b1f4c457b97adae52f9e1932cf21532d.jpg'
    foto_perfil2 = 'https://i.pinimg.com/736x/52/f7/d7/52f7d7db7b0d3e366bbb8ca60c289008.jpg'

    create_user('juan', 'juan', 'juan', foto_perfil1)
    create_user('pepe', 'pepe', 'pepe', foto_perfil2)
    create_user('mara', 'mara', 'mara', 'nada')

    user_id1 = get_uid('juan')
    user_id2 = get_uid('pepe')
    user_id3 = get_uid('mara')

    create_poem(user_id1, 'Poema de amor', poema1, 'Poema de amor')
    create_poem(user_id1, 'Poema de amor', poema2, 'Poema de amor')
    create_poem(user_id1, 'Poema de amor', poema3, 'Poema de amor')
    create_poem(user_id1, 'Poema de amor', poema4, 'Poema de amor')
    create_poem(user_id1, 'Poema de amor', poema5, 'Poema de amor')

    create_poem(user_id2, 'Poema de amor', poema1, 'Poema de amor')
    create_poem(user_id2, 'Poema de amor', poema2, 'Poema de amor')
    create_poem(user_id2, 'Poema de amor', poema3, 'Poema de amor')
    create_poem(user_id2, 'Poema de amor', poema4, 'Poema de amor')
    create_poem(user_id2, 'Poema de amor', poema5, 'Poema de amor')

    create_poem(user_id3, 'Poema de amor', poema1, 'Poema de amor')
    create_poem(user_id3, 'Poema de amor', poema2, 'Poema de amor')
    create_poem(user_id3, 'Poema de amor', poema3, 'Poema de amor')
    create_poem(user_id3, 'Poema de amor', poema4, 'Poema de amor')
    create_poem(user_id3, 'sin foto', poema5, 'Poema de amor')


    print("obtener la info del user1")
    get_user_info(user_id1)
    print("obtener la info del user2")
    get_user_info(user_id2)



