from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Cargar el modelo Keras previamente entrenado
model = load_model('ModelAI.h5')

# Función para preprocesar la imagen
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))  # Ajustar tamaño según sea necesario para tu modelo
    img = img / 255.0  # Normalizar píxeles a valores entre 0 y 1
    img = np.expand_dims(img, axis=0)  # Agregar dimensión del lote
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            try:
                img = preprocess_image(filename)
                print(f"Imagen preprocesada: {img.shape}")  # Depuración
                predictions = model.predict(img)
                class_index = np.argmax(predictions[0])
                classes = ['No es un incendio', 'Es un incendio']  # Define las clases según tu problema
                prediction = classes[class_index]
                return render_template('index.html', prediction=prediction, image_url=filename)
            except Exception as e:
                print(f"Error durante la predicción: {e}")  # Depuración
                return render_template('index.html', prediction=f"Error: {str(e)}", image_url=filename)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
		app.run(host='0.0.0.0', port=80)
