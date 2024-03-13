import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

num_clases = len(clases_calzados)
img_rows,img_cols = 64,64
#TODO Preguntar del tamaño de imagenes "img_rows,img_cols = 64,64"

def cargar_datos():
    data = []
    target = [] #Etiquetas

    for index, clase in enumerate(clases_calzados):
        folder_path  = folder_path = os.path.join('Entrenamiento', clase)

        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path,img)
            image = cv2.imread( img_path , cv2.IMREAD_GRAYSCALE )
            image = cv2.resize( image , (img_rows , img_cols) )
            data.append( np.array( image ) )
            target.append( index )

    data = np.array(data)
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    target = np.array(target)

    #NUMERO DE CLASES A DECLARAR
    new_target = to_categorical(target, num_clases)
    return data, new_target

data, target = cargar_datos()

x_train,x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_rows, img_cols,1)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #evita el sobre ajuste
model.add(Dropout(0.25)) #se da un porcentaje para evitar el sobre ajuste
model.add(Flatten()) #convierte los datos en una capa densa
model.add(Dense(128, activation='relu')) #128 neuronas
model.add(Dropout(0.5))
model.add(Dense(num_clases, activation='softmax')) #convierte los valores de la ultima capa con los valores de las clases

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train,y_train, batch_size=40, epochs=500, verbose=1, validation_data=(x_test,y_test))


#Guardo modelo / Entrenamiento
model.save('modelo.h5')

if not os.path.exists('graficas'):
    os.makedirs('graficas')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=clases_calzados, yticklabels=clases_calzados)
plt.xlabel('Prediccion')
plt.ylabel('Real')
plt.savefig('graficas/matriz_confusion.png')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Historial de error')
plt.ylabel("Error")
plt.xlabel("Epoca")
plt.legend(['Entrenamiento','Validacion'],loc="upper right")
plt.savefig("graficas/historial_error.png")
plt.show()