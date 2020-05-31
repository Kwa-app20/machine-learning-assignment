import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

from sklearn.datasets import load_boston
boston = load_boston()

print(boston.DESCR)

boston.feature_names

df = pd.DataFrame(boston.data, columns=boston.feature_names)

df.head()

df['MEDV'] = boston.target

df.head()

df.info()

df.describe()

sns.pairplot(df)

rows = 7
cols = 2

fig, ax = plt.subplots(nrows= rows, ncols= cols, figsize = (16,16))

col = df.columns
index = 0

for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax = ax[i][j])
        index = index + 1
        
plt.tight_layout()

fig, ax = plt.subplots(figsize = (16, 9))
sns.heatmap(df.corr(), annot = True, annot_kws={'size': 12})

def getCorrelatedFeature(corrdata, threshold):
       feature = []
       value = []
    
       for i, index in enumerate(corrdata.index):
           if abs(corrdata[index])> threshold:
               feature.append(index)
               value.append(corrdata[index])

       df = pd.DataFrame(data = value, index = feature, columns=['Corr Value'])
       return df

threshold = 0.4
corr_value = getCorrelatedFeature(df.corr()['MEDV'], threshold)

corr_value.index.values

correlated_data = df[corr_value.index]
correlated_data.head()

X = correlated_data.drop(labels=['MEDV'], axis = 1)
y = correlated_data['MEDV']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
→random_state=1)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50)

lm.intercept_

lm.coef_

def lin_func(values, coefficients=lm.coef_, y_axis=lm.intercept_):
    return np.dot(values, coefficients) + y_axis

from random import randint
for i in range(5):
index = randint(0,len(df)-1)
sample = df.iloc[index][corr_value.index.values].drop('MEDV')
print(
'PREDICTION: ', round(lin_func(sample),2),
' // REAL: ',df.iloc[index]['MEDV'],
' // DIFFERENCE: ', round(round(lin_func(sample),2) - df.
,→iloc[index]['MEDV'],2)
)


```python

```
