import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tensorflow.keras.models import load_model
import tensorflow as tf

from funciones import load_and_convert_image, get_date_time_from_image

def browse_folder():
    global ruta
    ruta = filedialog.askdirectory()
    ruta_entry.delete(0, tk.END)
    ruta_entry.insert(0, ruta)

def run_script():
    if not ruta:
        messagebox.showerror("Error", "Por favor, selecciona una ruta válida.")
        return

    data = []

    for root, dirs, files in os.walk(ruta):
        for file in files:
            if file.endswith('.JPG'):
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)

                sitio = parts[-4] if len(parts) > 4 else None
                año = parts[-3] if len(parts) > 3 else None
                camara = parts[-2] if len(parts) > 2 else None
                archivo = file
                extra = None

                if len(parts) > 5:
                    camara = parts[-3]
                    extra = parts[-2]

                data.append([full_path, sitio, año, camara, extra, archivo])

    df = pd.DataFrame(data, columns=['Ruta', 'Sitio', 'Año', 'Camara', 'Extra', 'Archivo'])

    df['Fecha'], df['Hora'] = zip(*df['Ruta'].apply(get_date_time_from_image))
    df['Fecha'] = pd.to_datetime(df['Fecha'], format="%Y:%m:%d")
    df['Hora'] = pd.to_datetime(df['Hora'], format="%H:%M:%S").dt.strftime("%H:%M:%S")

    image_tensors = [load_and_convert_image(img_path) for img_path in df['Ruta']]
    tensor = tf.stack(image_tensors)

    model = load_model('modeloAnimalVGG16.h5')

    df['Animal_proba'] = model.predict(tensor)

    df['Animal'] = (df['Animal_proba'] > 0.5).astype(int)

    indices = [i for i, x in enumerate(df['Animal'].values) if x == 1]

    tensorAnimal = tf.gather(tensor, indices)

    model = load_model('modeloGuanacoVGG16.h5')

    df.loc[df['Animal']==1, 'Guanaco_proba'] = model.predict(tensorAnimal)

    df.loc[df['Animal']==1,'Guanaco'] = (df.loc[df['Animal']==1,'Guanaco_proba'] > 0.5).astype(int)

    confianzaAnimal = 0.99
    confianzaGuanaco = 0.90

    df['Validar'] = ((df['Animal_proba'] >= (1-confianzaAnimal)) & (df['Animal_proba'] <= confianzaAnimal)) | (df['Guanaco_proba'] <= (confianzaGuanaco))

    df_a_validar = df[df['Validar']==True]
    df_animales = df[(df['Animal']==1) & (df['Validar']==False)]
    df_no_animales = df[(df['Animal']==0) & (df['Validar']==False)]

    df_a_validar.to_csv("validar.csv")
    df_animales.to_csv("animales.csv")
    df_no_animales.to_csv("no animales.csv")

    messagebox.showinfo("Terminado", "El script se ha ejecutado exitosamente.")

# Crear la ventana principal
root = tk.Tk()
root.title("Ejecutar Script")

ruta_label = tk.Label(root, text="Ruta de las imágenes:")
ruta_label.pack()

ruta_entry = tk.Entry(root, width=40)
ruta_entry.pack()

browse_button = tk.Button(root, text="Seleccionar Ruta", command=browse_folder)
browse_button.pack()

run_button = tk.Button(root, text="Ejecutar Script", command=run_script)
run_button.pack()

root.mainloop()
