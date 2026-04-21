# TUTORIEL COMPLET
# Reconnaissance de Chiffres Manuscrits avec TensorFlow et TensorFlow Lite

---

## 📚 À propos de ce tutoriel

Ce document présente un projet complet de machine learning pour la reconnaissance de chiffres manuscrits (0-9) utilisant TensorFlow et TensorFlow Lite. Vous apprendrez à entraîner un modèle CNN (Convolutional Neural Network - Réseau de Neurones Convolutif), à le convertir pour une utilisation mobile, et à créer une interface graphique interactive.

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Prérequis et installation](#2-prérequis-et-installation)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Entraînement du modèle CNN](#4-entraînement-du-modèle-cnn)
5. [Conversion en TensorFlow Lite](#5-conversion-en-tensorflow-lite)
6. [Test et inférence](#6-test-et-inférence)
7. [Interface graphique interactive](#7-interface-graphique-interactive)
8. [Résolution des problèmes courants](#8-résolution-des-problèmes-courants)
9. [Optimisations et améliorations possibles](#9-optimisations-et-améliorations-possibles)

---

## 1. Vue d'ensemble du projet

### Objectif

Ce projet implémente un système complet de reconnaissance de chiffres manuscrits capable de :

- Entraîner un réseau de neurones convolutif (CNN) sur le dataset MNIST (Modified National Institute of Standards and Technology)
- Atteindre une précision supérieure à 99% sur les données de test
- Convertir le modèle en format TensorFlow Lite pour un déploiement optimisé
- Fournir une interface graphique pour dessiner et reconnaître des chiffres en temps réel

### Technologies utilisées

| Technologie | Utilisation |
|------------|-------------|
| **TensorFlow/Keras** | Framework de deep learning pour l'entraînement du modèle |
| **TensorFlow Lite** | Format optimisé pour l'inférence rapide et le déploiement mobile |
| **NumPy** | Manipulation de matrices et preprocessing des données |
| **Tkinter** | Interface graphique pour le dessin et la reconnaissance |
| **Pillow (PIL)** | Traitement d'images (redimensionnement, filtres) |

---

## 2. Prérequis et installation

### Configuration système requise

- Python 3.8 ou supérieur
- 4 GB de RAM minimum (8 GB recommandé)
  - *Justification : Dataset MNIST (~50MB) + TensorFlow en mémoire (~2-3GB) + processus d'entraînement*
- 1 GB d'espace disque disponible
  - *Justification : TensorFlow + dépendances (~500-800MB) + dataset + modèles générés*
- GPU optionnel (mais accélère considérablement l'entraînement)
  - *Note : TensorFlow détecte automatiquement le GPU - aucune modification de code nécessaire*

### Installation des dépendances

**Recommandation importante** : Utilisez un environnement virtuel pour ne pas polluer votre système :

```bash
# Créer un environnement virtuel
python -m venv mnist_env

# Activer l'environnement virtuel
# Sur Linux/macOS :
source mnist_env/bin/activate

# Sur Windows :
mnist_env\Scripts\activate
```

Une fois l'environnement activé, installez les bibliothèques Python nécessaires :

```bash
pip install tensorflow numpy pillow
```

> 💡 **Note importante** : Tkinter est généralement inclus avec Python. Si vous rencontrez une erreur, installez-le avec :
> 
> ```bash
> # Linux (Ubuntu/Debian)
> sudo apt-get install python3-tk
> 
> # macOS (via Homebrew)
> brew install python-tk
> 
> # Windows : Tkinter est préinstallé avec Python
> # Si absent, réinstallez Python depuis python.org en cochant "tcl/tk and IDLE"
> ```

### Vérification de l'installation

Vérifiez que TensorFlow est correctement installé :

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Vérification du GPU (optionnel)** :

Si vous avez un GPU NVIDIA avec les drivers CUDA installés, vérifiez que TensorFlow le détecte :

```python
import tensorflow as tf

print("Version TensorFlow:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

# Si un GPU est détecté, vous verrez quelque chose comme :
# GPUs disponibles: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# Si aucun GPU n'est détecté :
# GPUs disponibles: []
```

> 💡 **Note** : TensorFlow utilisera automatiquement le GPU si disponible. Aucune modification du code n'est nécessaire ! L'entraînement sera 5-10x plus rapide avec un GPU.

---

## 3. Architecture du projet

### Structure des fichiers

```
mnist_project/
│
├── train_mnist_cnn.py       # Entraînement du modèle CNN
├── convert_to_tflite.py     # Conversion en TensorFlow Lite
├── test_tflite_host.py      # Test d'inférence
├── draw_and_predict.py      # Interface graphique
│
├── mnist_cnn_model/         # Modèle SavedModel (généré)
└── mnist_cnn_model.tflite   # Modèle TFLite (généré)
```

### Architecture du réseau de neurones

| Couche | Type | Configuration |
|--------|------|---------------|
| **Input** | Image | 28x28x1 (niveaux de gris) |
| **Conv2D 1** | Convolution | 32 filtres 3x3, activation ReLU |
| **MaxPool 1** | Max Pooling | Pool 2x2 |
| **Conv2D 2** | Convolution | 64 filtres 3x3, activation ReLU |
| **MaxPool 2** | Max Pooling | Pool 2x2 |
| **Flatten** | Aplatissement | Conversion en vecteur 1D |
| **Dense 1** | Couche dense | 128 neurones, activation ReLU |
| **Output** | Couche de sortie | 10 classes (0-9), softmax |

**Nombre total de paramètres** : ~100,000

---

## 4. Entraînement du modèle CNN

### Fichier : `train_mnist_cnn.py`

#### 4.1 Chargement et préparation des données

Le dataset MNIST contient 70 000 images de chiffres manuscrits :
- 60 000 images d'entraînement
- 10 000 images de test
- Chaque image : 28x28 pixels en niveaux de gris

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation (0-255 → 0-1)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Reshape pour CNN (ajouter dimension canal)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

#### 4.2 Augmentation des données

L'augmentation de données améliore la généralisation du modèle en créant des variations artificielles :

```python
# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,        # Rotation jusqu'à ±10 degrés
    width_shift_range=0.1,    # Décalage horizontal ±10%
    height_shift_range=0.1    # Décalage vertical ±10%
)
datagen.fit(x_train)
```

**Pourquoi l'augmentation de données ?**
- Réduit l'overfitting (surapprentissage)
- Améliore la robustesse du modèle
- Simule des variations réelles d'écriture
- Augmente artificiellement la taille du dataset

#### 4.3 Construction du modèle

```python
# Build CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Explication des couches** :
- **Conv2D** : Extrait des features (contours, formes) des images
- **MaxPooling** : Réduit la dimensionnalité et conserve les features importantes
- **Flatten** : Transforme la matrice 2D en vecteur 1D
- **Dense** : Couches fully connected pour la classification

#### 4.4 Entraînement

```python
print("Starting training...")
# Train with augmentation
model.fit(
    datagen.flow(x_train, y_train, batch_size=64), 
    epochs=12, 
    validation_data=(x_test, y_test)
)

# Évaluation finale
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

> ✅ **Résultats attendus** : Après 12 époques, le modèle devrait atteindre une précision de test d'environ **99.2% à 99.4%**.

**Exemple de sortie d'entraînement** :
```
Epoch 1/12
938/938 [==============================] - 15s 16ms/step - loss: 0.1234 - accuracy: 0.9621 - val_loss: 0.0456 - val_accuracy: 0.9856
Epoch 2/12
938/938 [==============================] - 14s 15ms/step - loss: 0.0523 - accuracy: 0.9842 - val_loss: 0.0321 - val_accuracy: 0.9892
...
Epoch 12/12
938/938 [==============================] - 14s 15ms/step - loss: 0.0098 - accuracy: 0.9971 - val_loss: 0.0234 - val_accuracy: 0.9934
Test accuracy: 0.9934
```

#### 4.5 Sauvegarde du modèle

```python
# Save as SavedModel
saved_model_dir = "mnist_cnn_model"
model.export(saved_model_dir)  # Keras 3 → proper SavedModel for TFLite
print(f"SavedModel exported to {saved_model_dir}")
```

**Important** : Utilisez `model.export()` et non `model.save()` pour Keras 3, afin de garantir la compatibilité avec TensorFlow Lite.

---

## 5. Conversion en TensorFlow Lite

### Pourquoi TensorFlow Lite ?

TensorFlow Lite est un format optimisé pour :
- **Réduire la taille du modèle** (compression)
- **Accélérer l'inférence** (optimisations)
- **Déploiement mobile** (Android, iOS) et edge devices
- **Réduire la consommation de mémoire** et d'énergie

### 5.1 Conversion de base

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open("mnist_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as 'mnist_cnn_model.tflite'")
```

### 5.2 Conversion avec quantification (optionnelle)

La quantification réduit encore plus la taille et accélère l'inférence :

```python
import tensorflow as tf

# Load saved model
converter = tf.lite.TFLiteConverter.from_saved_model("mnist_cnn_model")

# Enable optimization (basic quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save TFLite model
with open("mnist_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as 'mnist_cnn_model.tflite'")
```

### Comparaison des formats

| Type de conversion | Taille du modèle | Vitesse d'inférence | Précision |
|-------------------|------------------|---------------------|-----------|
| **Sans quantification** | ~400 KB | Baseline | 99.3% |
| **Quantification dynamique** | ~100 KB | 2-4x plus rapide | 99.2% |
| **Quantification INT8** | ~100 KB | 3-5x plus rapide | 99.0-99.2% |

---

## 6. Test et inférence

### Fichier : `test_tflite_host.py`

Ce script évalue le modèle TFLite sur de vraies images du jeu de test MNIST (précision + temps moyen d'inférence) :

```python
import numpy as np
import tensorflow as tf
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mnist_cnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load real MNIST test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # (N, 28, 28, 1)

# Keep it fast for a quick host-side check
NUM_SAMPLES = 1000
x_eval = x_test[:NUM_SAMPLES]
y_eval = y_test[:NUM_SAMPLES]

# Optional warm-up run
interpreter.set_tensor(input_details[0]['index'], x_eval[0:1])
interpreter.invoke()

correct = 0
total_inference_ms = 0.0

for i in range(NUM_SAMPLES):
    input_data = x_eval[i:i + 1]
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    end = time.perf_counter()

    pred = int(np.argmax(output))
    if pred == int(y_eval[i]):
        correct += 1
    total_inference_ms += (end - start) * 1000.0

accuracy = correct / NUM_SAMPLES
avg_inference_ms = total_inference_ms / NUM_SAMPLES

print(f"Samples tested: {NUM_SAMPLES}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Average inference time: {avg_inference_ms:.3f} ms/sample")
```

### Performances typiques

| Métrique | Valeur typique |
|-----------|----------------|
| **Accuracy (MNIST test)** | ~98.8% à 99.4% |
| **CPU (Intel i5), temps moyen** | 1-3 ms / image |
| **Mobile (ARM), temps moyen** | 5-15 ms / image |
| **GPU déléguée, temps moyen** | 0.5-1 ms / image |
| **Edge TPU, temps moyen** | < 0.5 ms / image |

> ⚡ **Note** : Les valeurs varient selon le matériel, la version TensorFlow/TFLite et le niveau de quantification.

---

## 7. Interface graphique interactive

### Fichier : `draw_and_predict.py`

Cette application Tkinter permet de dessiner des chiffres et d'obtenir des prédictions en temps réel.

### 7.1 Fonctionnalités principales

- Canvas de dessin 280x280 pixels avec trait lisse
- Prétraitement automatique de l'image (redimensionnement, inversion, flou)
- Prédiction instantanée avec affichage de la confiance
- Bouton de nettoyage pour recommencer
- Mesure du temps d'inférence

### 7.2 Pipeline de prétraitement

Pour que le modèle reconnaisse correctement les chiffres dessinés, plusieurs étapes de traitement sont nécessaires :

| Étape | Description |
|-------|-------------|
| **1. Resize** | Redimensionnement de 280x280 à 28x28 pixels (taille MNIST) |
| **2. Invert** | Inversion des couleurs (noir sur blanc → blanc sur noir comme MNIST) |
| **3. Blur** | Flou gaussien pour lisser les contours (réduit le bruit) |
| **4. Normalize** | Normalisation 0-255 → 0-1 (division par 255.0) |
| **5. Reshape** | Ajout de dimensions pour batch et canal : (1, 28, 28, 1) |

### 7.3 Code complet

```python
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
import time
import tensorflow as tf

# Load TFLite model
tflite_model_path = "mnist_cnn_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tkinter setup
WIDTH, HEIGHT = 280, 280  # drawing canvas size
window = tk.Tk()
window.title("Draw a digit (0-9)")

canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg="white")
canvas.grid(row=0, column=0, columnspan=4)

# PIL image to capture drawing
image = Image.new("L", (WIDTH, HEIGHT), color=255)
draw = ImageDraw.Draw(image)

last_x, last_y = None, None

def xy(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def add_line(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, x, y, width=15, fill="black", 
                          capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill=0, width=15)
    last_x, last_y = x, y

canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", add_line)

def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", (WIDTH, HEIGHT), color=255)
    draw = ImageDraw.Draw(image)

def predict_digit():
    # Resize, invert, blur, normalize
    img = image.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.GaussianBlur(1))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], arr)
    start = time.time()
    interpreter.invoke()
    end = time.time()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    pred = np.argmax(output)
    
    print(f"Prediction: {pred}")
    print(f"Confidence: {output[pred]:.4f}")
    print(f"Inference time: {(end-start)*1000:.2f} ms")

# Buttons
btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.grid(row=1, column=0, pady=5)

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.grid(row=1, column=1, pady=5)

window.mainloop()
```

### 7.4 Utilisation de l'interface

1. **Lancer l'application** : `python draw_and_predict.py`
2. **Dessiner un chiffre** : Cliquez et glissez sur le canvas blanc
3. **Prédire** : Cliquez sur le bouton "Predict"
4. **Voir les résultats** : La prédiction, la confiance et le temps d'inférence s'affichent dans la console
5. **Recommencer** : Cliquez sur "Clear" pour effacer le canvas

**Exemple de sortie console** :
```
Prediction: 7
Confidence: 0.9987
Inference time: 2.34 ms
```

---

## 8. Résolution des problèmes courants

### Problème 1 : Erreur lors de la conversion TFLite

**❌ Erreur** :
```
ValueError: Cannot convert a Keras model that has not been built
```

**✅ Solution** : Utilisez `model.export()` au lieu de `model.save()` pour Keras 3 :

```python
# ✅ Correct (Keras 3)
model.export("mnist_cnn_model")

# ❌ Ancien style (ne fonctionne pas toujours)
model.save("mnist_cnn_model")
```

---

### Problème 2 : Mauvaise précision avec l'interface graphique

**❌ Symptôme** : Le modèle prédit incorrectement les chiffres dessinés

**Causes possibles** :
- L'image n'est pas inversée (MNIST utilise du blanc sur fond noir)
- Problème de normalisation (valeurs doivent être entre 0 et 1)
- Mauvais reshape des dimensions
- Chiffre mal centré ou trop petit/grand

**✅ Solution** : Vérifiez le pipeline de prétraitement complet dans `predict_digit()` :

```python
# Pipeline complet requis
img = image.resize((28, 28), Image.Resampling.LANCZOS)  # 1. Resize
img = ImageOps.invert(img)                              # 2. Invert (CRUCIAL)
img = img.filter(ImageFilter.GaussianBlur(1))           # 3. Blur
arr = np.array(img, dtype=np.float32) / 255.0           # 4. Normalize
arr = arr.reshape(1, 28, 28, 1)                         # 5. Reshape
```

---

### Problème 3 : Fichier modèle introuvable

**❌ Erreur** :
```
FileNotFoundError: mnist_cnn_model.tflite
```

**✅ Solution** : Assurez-vous d'exécuter les scripts dans l'ordre :

1. **Entraînement** : `python train_mnist_cnn.py` (génère `mnist_cnn_model/` et `mnist_cnn_model.tflite`)
2. **Test** : `python test_tflite_host.py`
3. **Interface** : `python draw_and_predict.py`

---

### Problème 4 : Erreur de dépendance Tkinter

**❌ Erreur** :
```
ModuleNotFoundError: No module named 'tkinter'
```

**✅ Solution** : Installez Tkinter selon votre système :

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (via Homebrew)
brew install python-tk

# Windows : généralement préinstallé avec Python
# Si absent, réinstallez Python depuis python.org
```

---

### Problème 5 : Faible précision après entraînement

**❌ Symptôme** : Précision de test < 98%

**Causes possibles** :
- Nombre d'époques insuffisant
- Learning rate trop élevé ou trop bas
- Dataset mal chargé
- Pas d'augmentation de données

**✅ Solutions** :
1. Augmentez le nombre d'époques (12-15)
2. Ajustez le learning rate (essayez 0.0001 ou 0.01)
3. Vérifiez que le dataset est bien normalisé (0-1)
4. Utilisez l'augmentation de données

---

## 9. Optimisations et améliorations possibles

### 9.1 Ajouter du Dropout pour réduire l'overfitting

Le Dropout aide à éviter le surapprentissage en désactivant aléatoirement des neurones pendant l'entraînement :

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),  # ← Ajout
    
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),  # ← Ajout
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),   # ← Ajout (taux plus élevé sur Dense)
    keras.layers.Dense(10, activation='softmax')
])
```

**Avantages** :
- Réduit l'overfitting de 2-3%
- Améliore la généralisation
- Régularisation sans augmenter les paramètres

---

### 9.2 Utiliser des callbacks pour l'entraînement

Les callbacks permettent un contrôle avancé pendant l'entraînement :

```python
callbacks = [
    # Arrêt anticipé si la validation ne s'améliore plus
    keras.callbacks.EarlyStopping(
        patience=3, 
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    
    # Sauvegarder le meilleur modèle
    keras.callbacks.ModelCheckpoint(
        'best_model.keras', 
        save_best_only=True,
        monitor='val_accuracy'
    ),
    
    # Réduire le learning rate si stagnation
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, 
        patience=2,
        min_lr=0.00001
    )
]

model.fit(..., callbacks=callbacks)
```

---

### 9.3 Quantification INT8 complète

Pour une optimisation maximale sur mobile et edge devices :

```python
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force INT8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()

with open("mnist_cnn_model_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)
```

**Résultats** :
- Taille : ~80-100 KB (vs 400 KB non quantifié)
- Vitesse : 3-5x plus rapide
- Précision : ~98.8-99.1% (légère baisse acceptable)

---

### 9.4 Améliorer l'interface graphique

**Améliorations possibles** :

#### A. Afficher toutes les probabilités

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_probabilities(output):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(10), output)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=4)
```

#### B. Historique des prédictions

```python
history = []

def predict_digit():
    # ... code de prédiction ...
    history.append((pred, output[pred], (end-start)*1000))
    
    # Afficher les 5 dernières prédictions
    print("\n--- Historique ---")
    for i, (digit, conf, time_ms) in enumerate(history[-5:]):
        print(f"{i+1}. Digit: {digit}, Conf: {conf:.2%}, Time: {time_ms:.2f}ms")
```

#### C. Ajustement de l'épaisseur du trait

```python
brush_size = tk.IntVar(value=15)

def add_line(event):
    # Utiliser brush_size.get() au lieu de 15 en dur
    canvas.create_line(..., width=brush_size.get(), ...)
    draw.line(..., width=brush_size.get())

# Slider pour ajuster l'épaisseur
slider = tk.Scale(window, from_=5, to=30, orient=tk.HORIZONTAL, 
                  variable=brush_size, label="Brush Size")
slider.grid(row=1, column=2, pady=5)
```

#### D. Sauvegarde des dessins

```python
def save_drawing():
    filename = f"drawing_{int(time.time())}.png"
    image.save(filename)
    print(f"Saved as {filename}")

btn_save = tk.Button(window, text="Save Drawing", command=save_drawing)
btn_save.grid(row=1, column=3, pady=5)
```

---

### 9.5 Déploiement mobile

Le modèle TFLite peut être intégré dans des applications Android ou iOS :

#### A. Android (Java/Kotlin)

```kotlin
// Charger le modèle
val tflite = Interpreter(loadModelFile())

// Préparer l'input
val inputArray = arrayOf(floatArrayOf(...))
val outputArray = Array(1) { FloatArray(10) }

// Inférence
tflite.run(inputArray, outputArray)
val prediction = outputArray[0].indices.maxByOrNull { outputArray[0][it] }
```

#### B. iOS (Swift)

```swift
// Charger le modèle
guard let model = try? MNISTModel(configuration: MLModelConfiguration()) else {
    return
}

// Préparer l'input
let input = MNISTModelInput(image: processedImage)

// Inférence
guard let prediction = try? model.prediction(input: input) else {
    return
}
print("Predicted digit: \(prediction.classLabel)")
```

#### C. Flutter (Cross-platform)

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

// Charger le modèle
final interpreter = await Interpreter.fromAsset('mnist_cnn_model.tflite');

// Inférence
var input = [...]; // Processed image data
var output = List.filled(10, 0).reshape([1, 10]);
interpreter.run(input, output);
```

---

### 9.6 Batch Prediction pour plusieurs images

Pour traiter plusieurs images simultanément :

```python
def batch_predict(images_list):
    """
    images_list: List of PIL Images
    Returns: List of (prediction, confidence) tuples
    """
    # Prétraiter toutes les images
    batch = []
    for img in images_list:
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(1))
        arr = np.array(img, dtype=np.float32) / 255.0
        batch.append(arr)
    
    batch = np.array(batch).reshape(-1, 28, 28, 1)
    
    # Inférence batch
    interpreter.resize_tensor_input(input_details[0]['index'], batch.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], batch)
    interpreter.invoke()
    
    outputs = interpreter.get_tensor(output_details[0]['index'])
    
    results = []
    for output in outputs:
        pred = np.argmax(output)
        conf = output[pred]
        results.append((pred, conf))
    
    return results
```

---

### 9.7 Augmentation de données avancée

Pour une meilleure généralisation, utilisez des augmentations plus sophistiquées :

```python
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomTranslation

# Utiliser des couches d'augmentation dans le modèle
data_augmentation = keras.Sequential([
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomTranslation(0.1, 0.1),
])

model = keras.Sequential([
    data_augmentation,  # Appliqué pendant l'entraînement
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    # ... reste du modèle
])
```

---

### 9.8 Transfer Learning avec un modèle pré-entraîné

Pour des datasets plus complexes (comme EMNIST avec 47 classes) :

```python
# Charger un modèle pré-entraîné
base_model = keras.applications.MobileNetV2(
    input_shape=(28, 28, 1),
    include_top=False,
    weights=None
)

# Ajouter des couches personnalisées
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
```

---

## Conclusion

Ce projet démontre un pipeline complet de machine learning, de l'entraînement au déploiement. Vous avez appris à :

- ✅ Construire et entraîner un réseau de neurones convolutif avec TensorFlow/Keras
- ✅ Utiliser l'augmentation de données pour améliorer la généralisation
- ✅ Convertir un modèle en format TensorFlow Lite pour l'optimisation
- ✅ Créer une interface graphique interactive avec Tkinter
- ✅ Implémenter un pipeline de prétraitement d'images
- ✅ Résoudre les problèmes courants de déploiement
- ✅ Optimiser le modèle avec quantification et Dropout

### Prochaines étapes

1. **Expérimentez avec les hyperparamètres** : learning rate, nombre de filtres, époques
2. **Testez d'autres datasets** : EMNIST (lettres), Fashion-MNIST (vêtements)
3. **Déployez sur mobile** : Android/iOS app
4. **Créez une API REST** : Flask/FastAPI pour servir le modèle
5. **Ajoutez des métriques avancées** : matrice de confusion, courbes ROC
6. **Implémentez un système de logging** : TensorBoard pour visualiser l'entraînement

---

## 🎓 Ressources supplémentaires

### Documentation officielle
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Keras API Reference](https://keras.io/api/)

### Datasets
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [EMNIST (Extended MNIST)](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

### Tutoriels avancés
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [TensorFlow Lite for Mobile](https://www.tensorflow.org/lite/guide/android)
- [Post-training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

### Outils utiles
- **TensorBoard** : Visualisation de l'entraînement
- **Netron** : Visualiser l'architecture du modèle
- **WandB** : Tracking d'expériences ML

---

**Bon apprentissage ! 🚀**

*Ce tutoriel a été créé pour vous accompagner dans votre parcours de machine learning. N'hésitez pas à l'adapter et à l'améliorer selon vos besoins !*
