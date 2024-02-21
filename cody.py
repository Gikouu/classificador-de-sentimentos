import spacy
import string
import random
import pandas as pd
import seaborn as sns
import numpy as np

base_dados = pd.read_csv(r'C:\Users\giova\OneDrive\Documentos\Programas\Github\Classificador-de-Sentimentos\Train50.csv', encoding= 'utf-8')