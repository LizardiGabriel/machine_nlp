from pprint import pprint

import requests
import json

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
    else:
        print('Error creating user:', response.json())

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

# Crear usuarios de prueba
def test_create_users():
    create_user('juanito', 'juanito@gmail.com', 'juanito', 'https://i.pinimg.com/736x/b1/f4/c4/b1f4c457b97adae52f9e1932cf21532d.jpg')
    create_user('elizabeth', '3lizabeth37124@gmail.com', 'elizabeth', 'https://i.pinimg.com/736x/52/f7/d7/52f7d7db7b0d3e366bbb8ca60c289008.jpg')

# Crear poemas de prueba
def test_create_poems():

    poema1 = 'Nunca conocí la felicidad;\nNo creía que los sueños se hicieran realidad;\nNo podía creer realmente que estaba enamorado;\nHasta que finalmente te conocí'
    poema2 = 'Amarte no tiene fin ni principio;\nAmarte lo es todo;\nEs infinito en el tiempo;\nEs lo que siento por ti'

    create_poem(1, 'Poema de user1', poema1, 'me inspire hoy')
    create_poem(2, 'Poema de user2', poema2, 'plebes ya me perdieron')

# recuperar poemas
def get_poems():
    url = f'{BASE_URL}/getPoems'
    response = requests.get(url)
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=4, sort_keys=True))
    else:
        print('Error getting poems:', response.json())


# Ejecutar las pruebas
if __name__ == '__main__':
    test_create_users()
    test_create_poems()
    get_poems()
