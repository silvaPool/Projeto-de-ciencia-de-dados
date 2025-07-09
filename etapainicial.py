import pandas as pd

df = pd.read_excel("C:\projetoslivropython\default_of_credit_card_clients__courseware_version_1_21_19.xls")

#numero de linhas e colunas
print(df.shape)

#informações sobre as colunas
print(df.info())

#as primeiras linhas do DataFrame
print(df.head())

