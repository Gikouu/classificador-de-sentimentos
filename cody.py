#Importação de bibliotecas

import spacy
import string
import random
import pandas as pd
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.language import Language
from spacy_langdetect import LanguageDetector

#Criando as variáveis
#pln = processamento de linguagem natural

@Language.factory("categorias")
def get_lang_detector(nlp, name):
   return LanguageDetector()

base_dados = pd.read_csv(r'Train50.csv', encoding = 'utf-8', sep = ';')
pontuacoes = string.punctuation
stop_words = STOP_WORDS
base_dados_final = []
pln = spacy.load('pt_core_news_sm')
modelo = spacy.blank('pt')
categorias = modelo.create_pipe('textcat')
categorias.add_label('ALEGRIA')
categorias.add_label('MEDO')
modelo.add_pipe('categorias', last=True)
historico = []

#ajustando o banco de dados

def preprocessamento(texto):

    texto = texto.lower()
    documento = pln(texto)
    lista = []

    for token in documento:
        lista.append(token.lemma_)
    
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

    return lista

base_dados['tweet_text'] = base_dados['tweet_text'].apply(preprocessamento)

for texto, emocao in zip(base_dados['tweet_text'], base_dados['sentiment']):
    if emocao == 1:
        dic = ({'ALEGRIA':True, 'MEDO':False})
    elif emocao == 0:
        dic = ({'ALEGRIA':False, 'MEDO':True})
    
    base_dados_final.append([texto, dic.copy()])

#treinamento
    
modelo.begin_training()
for epoca in range(10):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final, 50):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses=losses)
        if epoca <= 10:
            print(losses)
            historico.append(losses)