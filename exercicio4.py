import pandas as pd

df = pd.read_excel("C:\projetoslivropython\default_of_credit_card_clients__courseware_version_1_21_19.xls")

#value_counts() listará os IDS exclusivos e a frequência com que ocorrem
id_counts = df['ID'].value_counts()
#print(id_counts.head())

#número de entradas duplicadas agrupadas executando outra contagem de valores
#print(id_counts.value_counts())

dupe_mask = id_counts == 2
#print(dupe_mask[0:5])

#print(id_counts.index[0:5])

#armazenar em uma nova variável os ids duplicados
dupe_ids = id_counts.index[dupe_mask]

#converter em uma lista e obter o seu tamanho
dupe_ids = list(dupe_ids)

#print(len(dupe_ids))

#print(dupe_ids[0:5])

#filtrando o dataframe para visualizar todas as colunas dos três primeiros IDs duplicados

#print(df.loc[df['ID'].isin(dupe_ids[0:3]),:].head(10))

#criando uma matriz booleana com mesmo tamanho do dataFrame
df_zero_mask = df == 0

#série booleana

feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)

#calcular a soma da série booleana
#print(sum(feature_zero_mask))

#limpando o dataframe eliminando as linhas só com zeros
df_clean_1 = df.loc[~feature_zero_mask,:].copy()

print(df_clean_1.shape)

print(df_clean_1['ID'].nunique())