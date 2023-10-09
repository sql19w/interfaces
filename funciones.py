import pandas as pd
import tensorflow as tf
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


# Suponiendo que df es tu DataFrame y 'Ruta' es la columna con las rutas de las imágenes
# Ejemplo:
# df = pd.DataFrame({'Ruta': ['ruta_imagen1.jpg', 'ruta_imagen2.jpg', ...]})

def load_and_convert_image(img_path):
    # Leer y decodificar la imagen
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # asumimos imágenes en color (3 canales)
    
    # Cambiar el tamaño de la imagen a 150x150
    img_resized = tf.image.resize(img, [150, 150])
    
    return tf.convert_to_tensor(img_resized)

def get_date_time_from_image(path):
    """Extrae la fecha y hora de la metadata de una imagen."""
    try:
        with Image.open(path) as img:
            exif_data = img._getexif()
            if exif_data and 36867 in exif_data:  # 36867 es el tag para DateTimeOriginal
                date_time = exif_data[36867]
                date, time = date_time.split(" ")
                return date, time
    except Exception as e:
        print(f"Error processing image {path}: {e}")

    return None, None