import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
url = 'https://raw.githubusercontent.com/Devika0901/codsoft/main/advertising.csv'
response = requests.get(url)
d = BytesIO(response.content)

d.head()
d.isnull().sum()
d.describe()
sns.pairplot(d, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=5, aspect=1, kind='scatter')
plt.show()
d.corr()
sns.heatmap(d.corr(),annot=True)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
X = d['TV']
y = d['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()
lr.summary()

