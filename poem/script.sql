-- Eliminar la base de datos si existe y crear una nueva
DROP DATABASE IF EXISTS poems;
CREATE DATABASE poems;
USE poems;

-- Tabla para almacenar los usuarios
CREATE TABLE Users (
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       user VARCHAR(255) NOT NULL,
                       correo VARCHAR(255) NOT NULL UNIQUE,
                       contra VARCHAR(255) NOT NULL,
                       foto_perfil_url VARCHAR(255),  -- Nueva columna para la foto de perfil
                       fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para almacenar las publicaciones de los usuarios
CREATE TABLE Posts (
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       user_id INT,
                       titulo VARCHAR(255) NOT NULL,
                       poema TEXT NOT NULL,
                       descripcion TEXT,
                       fecha_publicacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE
);

-- Tabla para almacenar los "me gusta" de los usuarios a las publicaciones
CREATE TABLE Likes (
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       user_id INT,
                       post_id INT,
                       fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE,
                       FOREIGN KEY (post_id) REFERENCES Posts(id) ON DELETE CASCADE,
                       UNIQUE(user_id, post_id)  -- Para asegurar que un usuario solo pueda dar un "me gusta" a una publicación
);

-- Tabla para almacenar los comentarios de los usuarios a las publicaciones
CREATE TABLE Comments (
                          id INT AUTO_INCREMENT PRIMARY KEY,
                          user_id INT,
                          post_id INT,
                          comentario TEXT NOT NULL,
                          fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE,
                          FOREIGN KEY (post_id) REFERENCES Posts(id) ON DELETE CASCADE
);

-- Tabla para gestionar el seguimiento entre usuarios (seguidores)
CREATE TABLE Followers (
                           id INT AUTO_INCREMENT PRIMARY KEY,
                           follower_id INT,  -- Usuario que sigue
                           following_id INT, -- Usuario al que se sigue
                           fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           FOREIGN KEY (follower_id) REFERENCES Users(id) ON DELETE CASCADE,
                           FOREIGN KEY (following_id) REFERENCES Users(id) ON DELETE CASCADE,
                           UNIQUE(follower_id, following_id)  -- Para asegurar que un usuario no pueda seguir a otro más de una vez
);

-- Consultas de prueba (ejemplo):
-- Mostrar todos los usuarios
SELECT * FROM Users;

-- Mostrar todas las publicaciones
SELECT * FROM Posts;

-- Mostrar los "me gusta" de una publicación específica
SELECT * FROM Likes WHERE post_id = 1;

-- Mostrar los comentarios de una publicación específica
SELECT * FROM Comments WHERE post_id = 1;

-- Mostrar todos los seguidores de un usuario
SELECT * FROM Followers WHERE following_id = 1;
