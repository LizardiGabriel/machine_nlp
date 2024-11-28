import pandas as pd
import re

# Cargar el archivo CSV
ruta_archivo = 'archivo.csv'
df = pd.read_csv(ruta_archivo)

# Cambiar el formato de la fecha a dd/mm/aaaa
df['Date'] = pd.to_datetime(df['Date'], format="%a, %d %b %Y %H:%M:%S %Z")
df['Date'] = df['Date'].dt.strftime("%d/%m/%Y")

# Función para asignar secciones basadas en el contenido de la URL
def asignar_seccion(row):
    # Si la sección es 'No section', buscar en el campo URL
    if row['Section'] == 'No section':
        url = row['URL']

        # Revisar palabras clave en la URL y asignar la sección correspondiente
        if re.search(r'/deportes/', url, re.IGNORECASE):
            return 'Deportes'
        elif re.search(r'/economia/', url, re.IGNORECASE):
            return 'Economía'
        elif re.search(r'/tecnologia/', url, re.IGNORECASE):
            return 'Tecnología'
        elif re.search(r'/cultura/', url, re.IGNORECASE):
            return 'Cultura'
        elif re.search(r'/opinion/', url, re.IGNORECASE):
            return 'Opinión'
        elif re.search(r'/ciencias/', url, re.IGNORECASE):
            return 'Ciencias'
    return row['Section']  # Mantener la sección actual si no es 'No section'

# Aplicar la función a la columna 'Section'
df['Section'] = df.apply(asignar_seccion, axis=1)

# Guardar el archivo actualizado si es necesario
df.to_csv('archivo_actualizado.csv', index=False)
print("Archivo procesado y guardado como 'archivo_actualizado.csv'")
