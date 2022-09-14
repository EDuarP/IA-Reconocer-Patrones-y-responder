from this import s
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import json
import random
import pickle
import tensorflow
import tflearn

with open ("contenido.json", encoding='utf-8') as archivo:
    datos=json.load(archivo)
   
palabras=[]
tags=[]
aux1=[]
aux2=[]

for contenido in datos["contenido"]:
     for patrones in contenido["patrones"]:
        auxPalabra = nltk.word_tokenize(patrones)
        palabras.extend(auxPalabra)
        aux1.append(auxPalabra)
        aux2.append(contenido["tag"])

        if contenido["tag"] not in tags:
            tags.append(contenido["tag"])

palabras = [stemmer.stem(w.lower()) for w in palabras if w!="?"] 
palabras = sorted(list(set(palabras)))
tags = sorted(tags)

entrenamiento = []
salida = []
salidaVacia = [0 for _ in range(len(tags))]

for x, documento in enumerate(aux1):
    cubeta = []
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
     if w in auxPalabra:
         cubeta.append(1)
     else:
         cubeta.append(0) 
    filaSalida = salidaVacia[:]
    filaSalida[tags.index(aux2[x])]=1
    entrenamiento.append(cubeta)
    salida.append(filaSalida)

entrenamiento = numpy.array(entrenamiento)
salida = numpy.array(salida)

tensorflow.compat.v1.reset_default_graph()

red= tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)
modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=10,show_metric=True)
modelo.save("modelo.tflearn")

def main():
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraIndivicial in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra==palabraIndivicial:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]

        print ("Sofia:", random.choice(respuesta))

main()
