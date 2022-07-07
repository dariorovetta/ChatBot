""" Paso 1. Importar bibliotecas y cargar los datos"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


intents_file = open("intents.json").read()
intents = json.loads(intents_file)

""" Paso 2. Preprocesamiento de los datos"""

# Tokenización

words = []
classes = []
documents = []
ignore_letters = ["!", "?", ",", "."]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenizar cada palabra
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Agregar documentos en el corpus
        documents.append((word, intent["tag"]))
        # Agregar a nuestra lista de clases
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

print(documents)

# Lematización

# Lemmaztize y baje cada palabra y elimine los duplicados

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in
         ignore_letters]
words = sorted(list(set(words)))

# Ordenar clases
classes = sorted(list(set(classes)))

# Documentos = combinación entre patrones e intenciones
print(len(documents), "documents")

# classes = intents
print(len(classes), "classes", classes)

# words = all words, vocabulary
print(len(words), "palabras lematizadas únicas", words)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

""" Paso 3. Crear datos de entrenamiento y prueba"""

# Crear los datos de entrenamiento
training = []

# Crear una matriz vacía para la salida
output_empty = [0] * len(classes)

# Conjunto de entrenamiento, bolsa de palabras para cada oración.
for doc in documents:
    bag = []  # Inicializando bolsa de palabras
    word_patterns = doc[0]  # Lista de palabras tokenizadas para el patrón.
    # Lematiza cada palabra - Crea una palabra base, en un intento
    # de representar palabras relacionadas.
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in
                     word_patterns]
    # Crea la matriz de bolsa de palabras con 1, si la palabra
    # se encuentra en el patrón actual.
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # La salida es un "0" para cada etiqueta y un "1" para la etiqueta actual.
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mezclar las características y hacer una matriz numpy.
random.shuffle(training)
training = np.array(training)

# Crear lista de entrenamiento y prueba. X: patrones. Y: intenciones
train_x = list(training[:, 0])
train_y = list(training[:, 1])

""" Paso 4. Entrenamiento del modelo"""

# Modelo de redes neuronales profundas
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Modelo de compilado. SGD con gradiente acelerado de Nesterov
# da buenos resultados.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])

# Entrenando y salvando el modelo
hist = model.fit(np.array(train_x), np.array(train_y, epochs=200,
                                             batch_size=5, verbose=1))
model.save("chatbot_model.h5", hist)

print("El modelo fue creado")
