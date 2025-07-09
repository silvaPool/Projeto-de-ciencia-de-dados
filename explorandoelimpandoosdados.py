import pandas as pd

df = pd.read_excel("C:\projetoslivropython\default_of_credit_card_clients__courseware_version_1_21_19.xls")

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

#print(df_clean_1.shape)

#print(df_clean_1['ID'].nunique())

#exercio novo abaixo

#print(df_clean_1.info())

#print(df_clean_1['PAY_1'].head(5))

#obtendo as contagens de valores da coluna PAY_1

#print(df_clean_1['PAY_1'].value_counts())

#encontrar todas as linhas que não tem dados ausentes para PAY_1

valid_pay_1_mask = df_clean_1['PAY_1'] != 'Not available'

#print(valid_pay_1_mask)

#print(sum(valid_pay_1_mask))

#limpando os dados eliminando as linhas de PAY_1

df_clean_2 = df_clean_1.loc[valid_pay_1_mask,:].copy()

#dimensão dos dados limpos
#print(df_clean_2.shape)

#converter o tipo de dado

df_clean_2['PAY_1'] = df_clean_2['PAY_1'].astype('int64')

#print(df_clean_2[['PAY_1', 'PAY_2']].info())

# EXERCICIO 6

import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 400

#print(df_clean_2[['LIMIT_BAL', 'AGE']].hist())

#relatório tabular de síntese estatística
#print(df_clean_2[['LIMIT_BAL', 'AGE']].describe())

#print(df_clean_2['EDUCATION'].value_counts())

#combinar os graus não documentados de EDUCATION com o grau outros

df_clean_2['EDUCATION'].replace(to_replace=[0,5,6], value=4, inplace=True)

#print(df_clean_2['EDUCATION'].value_counts())

#contagem de valores da caracteristica marriage

#print(df_clean_2['MARRIAGE'].value_counts())

#alterar os valores 0 para 3

df_clean_2['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)

#print(df_clean_2['MARRIAGE'].value_counts())

#implementando a OHE para uma característica categórica

#coluna vazia para os rótulos categóricos
df_clean_2['EDUCATION_CAT'] = 'none'

#print(df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10))

#crie um diconário que descreva o mapeamento das categorias de instrução

cat_mapping = {

    1: "graduate school",
    2: "university",
    3: "high school",
    4: "others"
    
}

df_clean_2['EDUCATION_CAT'] = df_clean_2['EDUCATION'].map(cat_mapping)

#print(df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10))

#novo DataFrame de codificação one-hot da coluna EDUCATION_CAT

edu_ohe = pd.get_dummies(df_clean_2['EDUCATION_CAT'], dtype=int)

#print(edu_ohe.head(10))

#Concatenação do DataFrame de codificação one-hot com o original

df_with_ohe = pd.concat([df_clean_2, edu_ohe], axis=1)

#print(df_with_ohe[['EDUCATION_CAT', 'graduate school', 'high school', 'university', 'others']].head(10))

#lista dos status de pagamento

#pay_feats = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# .describe para examinar sínteses estatísticas

#print(df[pay_feats].describe())

print(df['PAY_1'])