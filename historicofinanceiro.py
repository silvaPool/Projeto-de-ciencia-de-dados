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

#df_clean_2[pay_feats[0]].hist(bins=pay_1_bins)

#plt.xlabel('PAY_1')

#plt.ylabel('Number of accounts')

#plt.show()

#subplotagens

#mpl.rcParams['font.size'] = 4

#df_clean_2[pay_feats].hist(bins=pay_1_bins, layout=(2,3))

#plt.show()

# verificando os meses de agosto e julho

#print(df_clean_2.loc[df_clean_2['PAY_2'] == 2, ['PAY_2', 'PAY_3']].head())

#proporção da classe positiva
#print(df_clean_2['default payment next month'].mean())

#verificação do número de amostras em cada classe

#print(df_clean_2.groupby('default payment next month')['ID'].count())

from sklearn.linear_model import LogisticRegression

my_lr = LogisticRegression()

#print(my_lr)

my_new_lr = LogisticRegression(C=1.0, class_weight=None, dual=False,
                               fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='auto',
                               n_jobs=None, penalty='l2', random_state=None, solver='warn',
                               tol=0.0001, verbose=0, warm_start=False)

my_new_lr.C = 0.1
my_new_lr.solver = 'liblinear'

#print(my_new_lr)

# 10 primeiras amostras de uma única característica

X = df_clean_2['EDUCATION'][0:10].values.reshape(-1,1)

#print(X)

# 10 valores correspondentes da variável de resposta

Y = df_clean_2['default payment next month'][0:10].values

#print(Y)

# agora o objeto de modelo my_new_lr é um modelo treinado

my_new_lr.fit(X, Y)

# novas características para as quais serão feitas previsões

new_X = df_clean_2['EDUCATION'][10:20].values.reshape(-1,1)

#print(new_X)

# as previsões são feitas assim

#print(my_new_lr.predict(new_X))

# visualização de quais são os valores reais correspondentes a essas previsões

#print(df_clean_2['default payment next month'][10:20].values)

# Gerando dados sintéticos
np.random.seed(seed=1)

X = np.random.uniform(low=0.0, high=10.0, size=(1000,))

#print(X[0:10])

#criando dados da regressão linear

np.random.seed(seed=1)

#variável de inclinação

slope = 0.25

#variável de intercepção

intercept = -1.25

# variável de resposta

y = slope * X + np.random.normal(loc=0.0, scale=1.0, size=(1000,)) + intercept

mpl.rcParams['figure.dpi'] = 400
#plt.scatter(X, y, s=1)
#plt.show()

# Regressão linear com o Scikit-Learn

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X.reshape(-1,1), y)
#print(lin_reg.intercept_)
#print(lin_reg.coef_)

y_pred = lin_reg.predict(X.reshape(-1,1))

plt.scatter(X, y, s=1)
plt.plot(X, y_pred, 'r')
#plt.show()

# dados de treinamento e dados de teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_clean_2['EDUCATION'].values.reshape(-1,1), df_clean_2['default payment next month'].values,
    test_size=0.2, random_state=24)

#dimensão dos conjuntos de treinamento e teste

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# Frações das classes nos dados de treinamento e teste

#print(np.mean(y_train))
#print(np.mean(y_test))

# exemplo de modelo para ilustrar métricas de classificação binária

from sklearn.linear_model import LogisticRegression

example_lr = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                                intercept_scaling=1, max_iter=100, multi_class='auto',
                                n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
                                tol=0.0001, verbose=0, warm_start=False)

example_lr.fit(X_train, y_train)

y_pred = example_lr.predict(X_test)

# Calculando a acurácia da classificação com uma máscara lógica

is_correct = y_pred == y_test

#print(np.mean(is_correct))

# Calculando a acurácia da classificação com o scikit-learn

#print(example_lr.score(X_test, y_test))

from sklearn import metrics

#print(metrics.accuracy_score(y_test, y_pred))

#Calcular as taxas de verdadeiros e falsos positivos e negativos e a matriz de confusão

#número de amostras positivas

P = sum(y_test)

#print(P)

#número de verdadeiros positivos

TP = sum((y_test == 1) & (y_pred == 1))

#print(TP)

#Calculando a taxa de verdadeiros positivos

TPR = TP/P

#print(TPR)

#Falsos negativos

FN = sum((y_test == 1) & (y_pred == 0))

#print(FN)
 
#Taxa de falsos negativos

FNR = FN/P

#print(FNR)

#Calculando a TNR e a FPR

N = sum(y_test==0)
#print(N)

TN = sum((y_test == 0) & (y_pred == 0))
#print(TN)

FP = sum((y_test == 0) & (y_pred == 1))
#print(FP)

TNR = TN/N
FPR = FP/N
#print('The true negative rate is {} and the false positive rate is {}'.format(TNR, FPR))

#Matriz de confusão no scikit-learn

metrics.confusion_matrix(y_test, y_pred)

#Obtendo probabilidades previstas a partir de um modelo de regressão logística

#Obtendo as probabilidades previstas para as amostras de teste

y_pred_proba = example_lr.predict_proba(X_test)

#print(y_pred_proba)

#Soma das probabilidades previstas para cada amostra

prob_sum = np.sum(y_pred_proba,1)

#print(prob_sum)

#Verificando a forma do array

prob_sum.shape

#Elementos exclusivos do array

np.unique(prob_sum)

#Inserindo a segunda coluna do array de probabilidades previstas (probabilidade de associação de classe positiva) em um array

pos_proba = y_pred_proba[:,1]

#print(pos_proba)

mpl.rcParams['font.size'] = 12


#plt.hist(pos_proba)

#plt.xlabel('Predicted probability of positive class for testing data')

#plt.ylabel('Number pf samples')

#plt.show()

#Isolando as probabilidades previstas das amostras positivas e negativas

pos_sample_pos_proba = pos_proba[y_test==1]
neg_sample_pos_proba = pos_proba[y_test==0]

#Plotando um histograma empilhado

#plt.hist([pos_sample_pos_proba, neg_sample_pos_proba], histtype='barstacked')
#plt.legend(['Positive samples', 'Negative samples'])
#plt.xlabel('Predicted probability of positive class')
#plt.ylabel('Number of samples')
#plt.show()

# curva ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, pos_proba)

plt.plot(fpr, tpr, '*-')
plt.plot([0, 1], [0, 1], 'r--')
plt.legend(['Logistic regression', 'Random chance'])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')

plt.show()

print(thresholds)

print(metrics.roc_auc_score(y_test, pos_proba))



