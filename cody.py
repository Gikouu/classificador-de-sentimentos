#Importação de bibliotecas

import spacy
import string
import random
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from spacy.training import Example
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Criando as variáveis
#pln = processamento de linguagem natural

def get_lang_detector(nlp, name):
   return LanguageDetector()

base_dados = pd.read_csv(r'Train50.csv', encoding = 'utf-8', sep = ';')
base_teste = pd.read_csv(r'Test.csv', encoding = 'utf-8', sep = ';')
pontuacoes = string.punctuation
stop_words = STOP_WORDS
base_dados_final = []
pln = spacy.load('pt_core_news_sm')
modelo = spacy.blank('pt')
categorias = modelo.add_pipe("textcat")
categorias.add_label('ALEGRIA')
categorias.add_label('MEDO')
historico = []
historico_loss = []
previsoes = []
previsoes_final = []

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
base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)

for texto, emocao in zip(base_dados['tweet_text'], base_dados['sentiment']):
    if emocao == 1:
        dic = ({'ALEGRIA':True, 'MEDO':False})
    elif emocao == 0:
        dic = ({'ALEGRIA':False, 'MEDO':True})
    
    base_dados_final.append([texto, dic.copy()])

#treinamento
    
modelo.begin_training()
for epoca in range(5):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final, 512):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(
        textos, annotations
        )]
        modelo.update(examples, losses=losses)
        historico.append(losses)
        if epoca % 5 == 0:
            print(losses)
            historico.append(losses)

for i in historico:
  historico_loss.append(i.get('textcat'))
historico_loss = np.array(historico_loss)

modelo.to_disk('modelo')

#agora utilizamos a base de teste

for texto in base_teste['tweet_text']:
  previsao = modelo(texto)
  previsoes.append(previsao.cats)

for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
    previsoes_final.append(1)
  else:
    previsoes_final.append(0)

previsoes_final = np.array(previsoes_final)
respostas_reais = base_teste['sentiment'].values
cm = confusion_matrix(respostas_reais, previsoes_final)

#print dos resultados

print(cm)
print(accuracy_score(respostas_reais, previsoes_final))