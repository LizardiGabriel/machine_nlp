# importar un modulo de otro archivo de este paquete

from test import *



def init():
    drop_db()

    # crear un usuario
    juan_id = set_user(
        'juan',
        'juan',
        'juan',
        'juan',
        datetime.now().strftime('%d/%m/%y - %H:%M'),

    )


    david_id = set_user(
        'david',
        'david',
        'david',
        'david',
        '02/02/2021 - 12:00'
    )

    gabriel_id = set_user(
        'gabriel',
        'gabriel',
        'gabriel',
        'gabriel',
        '23/02/2021 - 12:00'
    )

    pepe_id = set_user(
        'pepe',
        'pepe',
        'pepe',
        'pepe',
        '4/03/2021 - 12:00'
    )

    # juan sigue a david
    user_follow(juan_id, david_id)
    user_follow(david_id, gabriel_id)


    juan_poem_id = set_poem(
        juan_id,
        'poema de juan',
        'este es el poema de juan',
        'poema de juan',
        datetime.now().strftime('%d/%m/%y - %H:%M')
    )

    #david crea un poema
    david_poem_id = set_poem(
        david_id,
        'poema de david',
        'este es el poema de david',
        'poema de david',
        datetime.now().strftime('%d/%m/%y - %H:%M')
    )

    # gabriel crea un poema
    gabriel_poem_id = set_poem(
        gabriel_id,
        'poema de gabriel',
        'este es el poema de gabriel',
        'poema de gabriel',
        datetime.now().strftime('%d/%m/%y - %H:%M')
    )

def imprimirUsuarios():
    print('Usuarios:')
    for user in get_users():
        print(user)
    print()

imprimirUsuarios()





