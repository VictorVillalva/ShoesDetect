import cv2
import numpy as np
from keras.models import load_model

model = load_model('modelo.h5')
image_path = input('Por favor, ingrese la ruta de la imagen : ')
image_path = image_path.strip('"')

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64,64))
img = np.array(img).reshape(-1,64,64,1)

prediction = model.predict(img)

clases_calzados = ['Futbol',
                   'Basquetball',
                   'Beisbol',
                   'Boxeo',
                   'Ciclismo',
                   'Futbol_Americano',
                   'Levantamiento_de_Pesas',
                   'Running',
                   'Senderismo',
                   'Tenis']
predicted_class = clases_calzados[np.argmax(prediction)]

print(f'El modelo del calzado es:  {predicted_class}')