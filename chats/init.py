import re

import spacy
#from spacy.tokenizer import Tokenizer
#from spacy.util import compile_infix_regex

nlp = spacy.blank("es")


def extraer_mensajes(remitente, contenido):
    mensajes = []

    patron_remitenteBueno = fr"\[\d{{2}}/\d{{2}}/\d{{2}}, \d{{1,2}}:\d{{2}}:\d{{2}} (?:a\.m\.|p\.m\.)\] {remitente} ?: (.*)"
    patron_remitenteMal1 = fr"\[\d{{2}}/\d{{2}}/\d{{2}}, \d{{1,2}}:\d{{2}}:\d{{2}} (?:a\.m\.|p\.m\.)\] {remitente} ?: ‎(.*)"
    patron_remitenteMal2 = fr"‎(.*)"

    patron_tiempo = r"\[\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2} (?:a\.m\.|p\.m\.)\]"

    mensaje_actual = ""

    for linea in contenido.splitlines():
        esmal2 = re.match(patron_remitenteMal1, linea)
        coincidencia = re.match(patron_remitenteBueno, linea)

        if esmal2:
            coincidencia = not esmal2

        if coincidencia:
            if mensaje_actual:
                mensajes.append(mensaje_actual.strip())

            mensaje_actual = coincidencia.group(1)

        elif mensaje_actual:
            if re.match(patron_remitenteMal2, linea):
                continue

            if not re.match(patron_tiempo, linea):
                mensajes.append(mensaje_actual.strip())
                mensaje_actual = linea.strip()

            else:
                mensajes.append(mensaje_actual.strip())
                mensaje_actual = ""

    if mensaje_actual:
        mensajes.append(mensaje_actual.strip())

    return mensajes

def quitarLRM(mensajes):
    patron_crudo = fr"(.*)‎<Se editó este mensaje.>(.*)"
    mensajes_limpios = []
    for mensaje in mensajes:
        coincidencia = re.match(patron_crudo, mensaje)
        if coincidencia:
            mensajes_limpios.append(coincidencia.group(1).strip())
        else:
            mensajes_limpios.append(mensaje.strip())

    return mensajes_limpios



def procesar_mensajes_spacy(mensajes):
    #infixes = nlp.Defaults.infixes + [r'(?<=producto):\d+-\d+']
    #infix_re = compile_infix_regex(infixes)
    #nlp.tokenizer.infix_finditer = infix_re.finditer

    processed_messages = []
    for mensaje in mensajes:
        doc = nlp(mensaje.strip())
        tokens = []
        for sentence in doc:
            tokens.append(sentence.text)

        processed_messages.append(f"<s> {' '.join(tokens)} </s>")

    return processed_messages


def extraer_remitentes(contenido):
    remitentes = set()
    patron_remitente = r"\[\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2} (?:a\.m\.|p\.m\.)\] ([\w\s]+) ?:"
    for linea in contenido.splitlines():
        coincidencia = re.match(patron_remitente, linea)
        if coincidencia:
            remitentes.add(coincidencia.group(1))
            if len(remitentes) == 2:
                break

    return remitentes



def procesarChat(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()

    remitentes = extraer_remitentes(contenido)
    rem1, rem2 = remitentes
    print(f"Remitentes: {remitentes}")

    mensajes_rem1 = extraer_mensajes(rem1, contenido)
    mensajes_rem2 = extraer_mensajes(rem2, contenido)

    mensajes_rem1 = quitarLRM(mensajes_rem1)
    mensajes_rem2 = quitarLRM(mensajes_rem2)

    mensajes_rem1_procesados = procesar_mensajes_spacy(mensajes_rem1)
    mensajes_rem2_procesados = procesar_mensajes_spacy(mensajes_rem2)

    ret = [f"./procesados/{rem1}-{rem2}.txt", f"./procesados/{rem2}-{rem1}.txt"]

    with open(ret[0], "w", encoding="utf-8") as archivo_rem1:
        archivo_rem1.write("\n".join(mensajes_rem1_procesados))

    with open(ret[1], "w", encoding="utf-8") as archivo_rem2:
        archivo_rem2.write("\n".join(mensajes_rem2_procesados))



    return ret



