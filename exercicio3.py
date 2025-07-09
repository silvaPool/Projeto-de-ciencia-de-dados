import pandas as pd

df = pd.read_excel("C:\projetoslivropython\default_of_credit_card_clients__courseware_version_1_21_19.xls")

#verificar se o numero de ids é igual ao numero de linhas

#nomes das colunas
print(df.columns)

print(df.head())

#numero de valores exclusivos
print(df['ID'].nunique())


print(df.shape)

#value_counts() listará os IDS exclusivos e a frequência com que ocorrem

id_counts = df['ID'].value_counts()
print(id_counts.head())

#número de entradas duplicadas agrupadas executando outra contagem de valores

print(id_counts.value_counts())




