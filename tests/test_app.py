import unittest
from flask import Flask, request, render_template, redirect, url_for
from flask_testing import TestCase
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from unittest.mock import patch, MagicMock

# Supongamos que el código Flask está en un archivo llamado app.py
from app import app

class FlaskAppTest(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = 'test/uploads/'
        return app

    def setUp(self):
        # Crear la carpeta de pruebas si no existe
        if not os.path.exists(self.app.config['UPLOAD_FOLDER']):
            os.makedirs(self.app.config['UPLOAD_FOLDER'])

    def tearDown(self):
        # Eliminar la carpeta de pruebas y su contenido
        if os.path.exists(self.app.config['UPLOAD_FOLDER']):
            for file in os.listdir(self.app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], file)
                os.remove(file_path)
            os.rmdir(self.app.config['UPLOAD_FOLDER'])

    @patch('app.load_model')
    @patch('app.cv2.imread')
    @patch('app.cv2.resize')
    def test_index_post(self, mock_resize, mock_imread, mock_load_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.9, 0.1]])
        mock_load_model.return_value = mock_model

        mock_imread.return_value = np.ones((64, 64, 3))
        mock_resize.return_value = np.ones((32, 32, 3))

        with open('test_image.jpg', 'wb') as f:
            f.write(os.urandom(1024))

        with open('test_image.jpg', 'rb') as img:
            data = {
                'file': (img, 'test_image.jpg')
            }
            response = self.client.post('/', data=data, content_type='multipart/form-data')
        
        os.remove('test_image.jpg')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Es un incendio', response.data)

    def test_index_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'prediction', response.data)

if __name__ == '__main__':
    unittest.main()
