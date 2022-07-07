""" Paso 5. Interactuando con el chat bot"""
# Importar las librerías
import pickle
import numpy as np
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from tkinter import *
model = load_model("chatbot_model.h5")
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


def clean_up_sentence(sentence):
    # Tokenizar el patrón: dividir palabras en una matriz
    sentence_words = nltk.word_tokenize(sentence)
    # Derivación de cada palabra - Reducción a la forma básica
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word
                      in sentence_words]
    return sentence_words
# Bolsa de retorno de la matriz de palabras: 0 o 1 para las
# palabras que existen en la oración


def bag_of_words(sentence, letter, show_details=True):
    # Patrones de tokenización
    sentence_words = clean_up_sentence(sentence)
    # Bolsa de palabras - Matriz de vocabulario
    bag = [0]*len(letter)
    for s in sentence_words:
        for i, word in enumerate(letter):
            if word == s:
                # Asignar 1 si la palabra actual está en la
                # posición de vocabulario
                bag[i] = 1
                if show_details:
                    print("Encontrado en la bolsa: %s" % word)
    return np.array(bag)


def predict_class(sentence):
    # Filtrar por debajo de las predicciones del umbral
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Probabilidad de fuerza de clasificación
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability":
                            str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


# Creando la GUI de TKinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Tú: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        ints = predict_class(msg)
        res = getResponse(ints, intents)

        ChatBox.insert(END, "Bot: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Crear ventana de chat
ChatBox = Text(root, bd=0, bg="white", height="8", width=50,
               font="Arial",)

ChatBox.config(state=DISABLED)

# Enlazar la barra de desplazamiento a la ventana del chat
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Crear botón para enviar mensaje
SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Enviar",
                    width="10", height=5, bd=0, bg="#f9a602",
                    activebackground="#3c9d9b", fg='#000000',
                    command=send)

# Crear el cuadro para ingresar el mensaje
EntryBox = Text(root, bd=0, bg="white", width=29, height="5",
                font="Arial")
EntryBox.bind("<Return>", send)

# Coloque todos los componentes en la pantalla
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()
