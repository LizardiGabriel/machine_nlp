# importar un modulo de otro archivo de este paquete


from datetime import datetime
from test import (
    drop_db, set_user, user_follow, set_poem, get_poems_by_followed_users
)

# 1. Borrar la base de datos
drop_db()

# 2. Crear 5 usuarios
user1_id = set_user('user1', 'user1@example.com', 'password123', 'url_foto_user1', datetime.now().strftime('%d/%m/%y - %H:%M'))
user2_id = set_user('user2', 'user2@example.com', 'password123', 'url_foto_user2', datetime.now().strftime('%d/%m/%y - %H:%M'))
user3_id = set_user('user3', 'user3@example.com', 'password123', 'url_foto_user3', datetime.now().strftime('%d/%m/%y - %H:%M'))
user4_id = set_user('user4', 'user4@example.com', 'password123', 'url_foto_user4', datetime.now().strftime('%d/%m/%y - %H:%M'))
user5_id = set_user('user5', 'user5@example.com', 'password123', 'url_foto_user5', datetime.now().strftime('%d/%m/%y - %H:%M'))

# 3. Crear una red de seguimiento entre los usuarios
user_follow(user1_id, user2_id)
user_follow(user1_id, user3_id)
user_follow(user2_id, user1_id)
user_follow(user2_id, user4_id)
user_follow(user3_id, user4_id)
user_follow(user4_id, user5_id)

# 4. Crear poemas para cada usuario
set_poem(user1_id, 'Poema de la Naturaleza', 'El viento susurra...', 'Un poema sobre la belleza del mundo natural', datetime.now().strftime('%d/%m/%y - %H:%M'))
set_poem(user2_id, 'Oda al Amor', 'En el jardín de mis sueños...', 'Un poema romántico', datetime.now().strftime('%d/%m/%y - %H:%M'))
set_poem(user3_id, 'La Ciudad', 'Entre el asfalto y el neón...', 'Un poema urbano', datetime.now().strftime('%d/%m/%y - %H:%M'))
set_poem(user4_id, 'Soledad', 'En la quietud de la noche...', 'Un poema reflexivo', datetime.now().strftime('%d/%m/%y - %H:%M'))
set_poem(user5_id, 'Alegría', 'El sol brilla...', 'Un poema optimista', datetime.now().strftime('%d/%m/%y - %H:%M'))

# 5. Obtener los poemas de las personas que sigue el usuario 1
poemas_seguidos = get_poems_by_followed_users(user1_id)

# Imprimir los poemas
for poema in poemas_seguidos:
    print(f"Título: {poema['titulo']}")
    print(f"Poema: {poema['poema']}")
    print(f"Descripción: {poema['descripcion']}")
    print(f"Fecha de publicación: {poema['fecha_publicacion']}")
    print(f"Usuario: {poema['usuario']}")
    print(f"Foto de perfil: {poema['foto_perfil_url']}")
    
    print("-" * 20)



