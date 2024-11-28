import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
import csv

bigramas = {}
trigramas = {}
tipo_ngrama = ""
palabras_iniciales = []
texto_generado_completo = []

def cargar_archivo():
    global bigramas, trigramas, tipo_ngrama
    bigramas.clear()
    trigramas.clear()

    file_path = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Leer la primera línea como encabezado

        if header[:3] == ["Term1", "Term2", "Term3"]:
            tipo_ngrama = "trigrama"
            for row in reader:
                term1, term2, term3, freq, context_freq, prob = row
                trigramas[(term1, term2, term3)] = float(prob)
        elif header[:2] == ["Term1", "Term2"]:
            tipo_ngrama = "bigrama"
            for row in reader:
                term1, term2, freq, context_freq, prob = row
                bigramas[(term1, term2)] = float(prob)
        else:
            messagebox.showerror("Error", "Formato de archivo no válido.")
            return

    messagebox.showinfo("Éxito", f"Archivo de {tipo_ngrama}s cargado correctamente.")
    ngram_label.config(text=f"Modelo de {tipo_ngrama}")

    # obtener nombre del archivo de la ruta
    nombre_archivo = file_path.split("/")[-1]

    entrada_archivo.delete(0, tk.END)
    entrada_archivo.insert(0, nombre_archivo)


def calcular_siguientes_palabras():
    entrada = texto_generado.get("1.0", tk.END).strip().split()

    if not entrada:
        messagebox.showwarning("Advertencia", "Escribe al menos una palabra para empezar.")
        return

    palabras_probables = []

    if tipo_ngrama == "bigrama" and len(entrada) >= 1:
        ultima_palabra = entrada[-1]
        palabras_probables = [(term2 if term2 != "</s>" else ".", prob) for (term1, term2), prob in bigramas.items() if term1 == ultima_palabra]

    elif tipo_ngrama == "trigrama":
        if len(entrada) == 1:
            entrada.insert(0, "<s>")

        if len(entrada) >= 2:
            penultima_palabra, ultima_palabra = entrada[-2], entrada[-1]
            palabras_probables = [(term3 if term3 != "</s>" else ".", prob) for (term1, term2, term3), prob in trigramas.items() if term1 == penultima_palabra and term2 == ultima_palabra]

    palabras_probables.sort(key=lambda x: x[1], reverse=True)
    print(entrada, "-->")
    print(palabras_probables)

    # 3 palabras mas probables
    mas_probables = palabras_probables[:3]

    # buscar si existe la palabra ".", si no existe ponerla en 4a opcion, si exite, nada
    if "." not in [palabra for palabra, _ in mas_probables]:
        mas_probables.append((".", 0))


    lista_sugerencias.delete(0, tk.END)
    for palabra, _ in mas_probables:
        lista_sugerencias.insert(tk.END, palabra)


def agregar_palabra():
    seleccion = lista_sugerencias.curselection()
    if seleccion:
        palabra = lista_sugerencias.get(seleccion)

        texto_generado_completo.append(palabra)
        texto_generado.delete("1.0", tk.END)
        texto_generado.insert(tk.END, " ".join(texto_generado_completo) + " ")

        # Calcular nuevas palabras tras agregar una
        calcular_siguientes_palabras()



root = tk.Tk()
root.title("Generador de Texto")

btn_cargar_archivo = tk.Button(root, text="Cargar archivo CSV", command=cargar_archivo)
btn_cargar_archivo.grid(row=0, column=0, pady=5)

entrada_archivo = tk.Entry(root)
entrada_archivo.config(width=30)
entrada_archivo.grid(row=0, column=1, pady=5)

ngram_label = tk.Label(root, text="Modelo de n-grama:")
ngram_label.grid(row=1, column=0, columnspan=2, pady=5)



placeholder_label = tk.Label(root, text="Palabras iniciales:")
placeholder_label.grid(row=2, column=0, sticky="e")
placeholder_entry = tk.Entry(root, width=30)
placeholder_entry.grid(row=2, column=1)

btn_calcular = tk.Button(root, text="Calcular siguientes palabras", command=calcular_siguientes_palabras)
btn_calcular.grid(row=3, column=0, columnspan=2, pady=5)

lista_sugerencias = Listbox(root, width=30, height=5)
lista_sugerencias.grid(row=4, column=0, columnspan=2, pady=5)

btn_agregar_palabra = tk.Button(root, text="Add word", command=agregar_palabra)
btn_agregar_palabra.grid(row=5, column=0, columnspan=2, pady=5)

texto_generado = tk.Text(root, width=40, height=5)
texto_generado.grid(row=6, column=0, columnspan=2, pady=5)

def establecer_palabras_iniciales():
    global palabras_iniciales, texto_generado_completo
    palabras_iniciales = placeholder_entry.get().strip().split()
    texto_generado_completo = palabras_iniciales.copy()
    texto_generado.delete("1.0", tk.END)
    texto_generado.insert(tk.END, " ".join(texto_generado_completo) + " ")
    calcular_siguientes_palabras()

btn_calcular.config(command=establecer_palabras_iniciales)
root.mainloop()
