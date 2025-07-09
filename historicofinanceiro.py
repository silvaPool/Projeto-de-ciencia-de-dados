import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 400

import numpy as np


df = pd.read_excel("C:\projetoslivropython\default_of_credit_card_clients__courseware_version_1_21_19.xls")

id_counts = df['ID'].value_counts()

dupe_mask = id_counts == 2

dupe_ids = id_counts.index[dupe_mask]

dupe_ids = list(dupe_ids)

df_zero_mask = df == 0

feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)

df_clean_1 = df.loc[~feature_zero_mask,:].copy()

valid_pay_1_mask = df_clean_1['PAY_1'] != 'Not available'

df_clean_2 = df_clean_1.loc[valid_pay_1_mask,:].copy()

df_clean_2['PAY_1'] = df_clean_2['PAY_1'].astype('int64')

#status de pagamento em uma lista

pay_feats = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# .describe para examinar sínteses estatísticas

#print(df[pay_feats].describe())

# examinar os valores de PAY_1

#print(df_clean_2[pay_feats[0]].value_counts().sort_index())

#df_clean_2[pay_feats[0]].hist()

#plt.show()

# array de 12 números que resultará em 11 bins

pay_1_bins = np.array(range(-2,10)) - 0.5

#print(pay_1_bins)

#adicionando rótulos aos eixos

df_clean_2[pay_feats[0]].hist(bins=pay_1_bins)

plt.xlabel('PAY_1')

plt.ylabel('Number of accounts')

#plt.show()

#subplotagens

mpl.rcParams['font.size'] = 4

df_clean_2[pay_feats].hist(bins=pay_1_bins, layout=(2,3))

#plt.show()

# verificando os meses de agosto e julho

print(df_clean_2.loc[df_clean_2['PAY_2'] == 2, ['PAY_2', 'PAY_3']].head())