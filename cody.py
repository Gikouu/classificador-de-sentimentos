#Importação de bibliotecas
import spacy
import string
import random
import pandas as pd
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS

#Criando as variáveis
base_dados = pd.read_csv(r'Train50.csv', encoding = 'utf-8', sep = ';')
pontuacoes = string.punctuation
stop_words = STOP_WORDS
base_dados_final = []

#pln = processamento de linguagem natural
pln = spacy.load('pt_core_news_sm')

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
        dic = ({'ALEGRIA':True, 'TRISTEZA':False})
    elif emocao == 0:
        dic = ({'ALEGRIA':False, 'TRISTEZA':True})
    
    base_dados_final.append([texto, dic.copy()])