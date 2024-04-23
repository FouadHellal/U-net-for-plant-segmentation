import tensorflow as tf
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3

def load_images_and_masks(base_dir, target_size=(224, 224)):
    images = []
    masks = []
    
    # Parcourir chaque sous-dossier dans le répertoire de base
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        # Trouver les fichiers d'image et de masque
        files = os.listdir(folder_path)
        image_file = next((f for f in files if f.endswith('.jpg')), None)
        mask_file = next((f for f in files if f.endswith('.png')), None)

        if image_file is None or mask_file is None:
            print(f"Missing files in {folder_path}")
            continue
        
        # Chemins complets pour l'image et le masque
        image_path = os.path.join(folder_path, image_file)
        mask_path = os.path.join(folder_path, mask_file)
        
        # Charger l'image et le masque
        image = imread(image_path) / 255.0
        mask = imread(mask_path, as_gray=True) / 255.0
        mask = mask > 0.5  # Binariser le masque
        
        # Redimensionner l'image et le masque
        image_resized = resize(image, target_size + (3,), preserve_range=True)
        mask_resized = resize(mask, target_size, mode='constant', preserve_range=True, anti_aliasing=False)
        mask_resized = np.expand_dims(mask_resized, axis=-1)  # Ajouter une dimension de canal au masque
        
        if tf.random.uniform(()) > 0.5:
            # Random flipping of the image and mask
            image_resized = tf.image.flip_left_right(image_resized)
            mask_resized = tf.image.flip_left_right(mask_resized)
        
        images.append(image_resized)
        masks.append(mask_resized)
    
    return np.array(images), np.array(masks)

# Utilisation de la fonction
base_dir = '/NEWUNET/'
x_train, y_train = load_images_and_masks(os.path.join(base_dir, 'train'))
x_val, y_val = load_images_and_masks(os.path.join(base_dir, 'val'))
x_test, y_test = load_images_and_masks(os.path.join(base_dir, 'test'))

print("Données d'entraînement :")
print("Images :", x_train.shape)
print("Masques :", y_train.shape)

print("\nDonnées de validation :")
print("Images :", x_val.shape)
print("Masques :", y_val.shape)

print("\nDonnées de test :")
print("Images :", x_test.shape)
print("Masques :", y_test.shape)



#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    batch_size=16,
                    epochs=100,
                    )

import matplotlib.pyplot as plt

def plot_model_performance(history):
    # Récupération des données d'accuracy et de loss pour l'entraînement et la validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # Création des graphiques pour l'accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Création des graphiques pour le loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Utilisation de la fonction avec l'historique de l'apprentissage de votre modèle
plot_model_performance(history)

# save the model as HDF5
model.save('modelunetcoffee.h5')

from tensorflow.keras.models import load_model

model = load_model('modelunetcoffee.h5')
import tensorflow as tf


# Évaluation du modèle sur les données de test
eval_result = model.evaluate(x_test, y_test)
print(f'Test Loss: {eval_result[0]*100}, Test Accuracy: {eval_result[1]*100}')




def load_and_prepare_image(image_path, target_size=(224, 224)):
    image = imread(image_path) / 255.0  # Charger et normaliser l'image
    image_resized = resize(image, target_size + (3,), preserve_range=True)  # Redimensionner l'image
    image_resized = np.expand_dims(image_resized, axis=0)  # Ajouter une dimension batch
    return image_resized

image_path = 'plant villag/image/30.85335078044677_39.jpg'  # Remplacez par le chemin vers votre image
image_to_predict = load_and_prepare_image(image_path)

predicted_mask = model.predict(image_to_predict)
predicted_mask = predicted_mask[0]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_to_predict[0])  # Afficher l'image originale
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(predicted_mask.squeeze(), cmap='gray')  # Afficher le masque prédit
ax[1].set_title('Predicted Mask')
ax[1].axis('off')

plt.show()

