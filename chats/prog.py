import tkinter as tk
from tkinter import filedialog, messagebox

import init as init
import ngram as ngram


def procesar_chat(nombre_archivo, n):
    mensajes1 = init.procesarChat(nombre_archivo)

    archivo1 = ngram.sacar_ngramas(mensajes1[0], n)
    archivo2 = ngram.sacar_ngramas(mensajes1[1], n)
    res_generados = f"{archivo1} y {archivo2}"
    return res_generados

def seleccionar_archivo():
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    entrada_archivo.delete(0, tk.END)
    entrada_archivo.insert(0, archivo)


def generar_ngramas(n):
    archivo = entrada_archivo.get()
    if archivo:
        impr = procesar_chat(archivo, n)
        messagebox.showinfo("Éxito", f"Ngramas generados en: {impr}")
    else:
        messagebox.showerror("Error", "Debe seleccionar un archivo de texto")


ventana = tk.Tk()
ventana.title("Cargador de Chats de WhatsApp")


etiqueta = tk.Label(ventana, text="Cargar chat de WhatsApp")
etiqueta.config(font=("Arial", 20), fg="blue")
etiqueta.pack()

entrada_archivo = tk.Entry(ventana)
entrada_archivo.config(width=100)
entrada_archivo.pack()

boton_explorar = tk.Button(ventana, text="Seleccionar archivo", command=seleccionar_archivo)
boton_explorar.pack()

boton_bigramas = tk.Button(ventana, text="Generar bigramas", command=lambda: generar_ngramas(2))
boton_bigramas.pack()

boton_trigramas = tk.Button(ventana, text="Generar trigramas", command=lambda: generar_ngramas(3))
boton_trigramas.pack()

ventana.mainloop()
