import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data=pd.read_csv("IRIS.csv")
data.head()
data.describe()
x=data.drop('species',axis=1)
y=data['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.transform(x_test)
def model_train_test(model):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(classification_report(y_test,y_pred))
model_train_test(RandomForestClassifier())
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))
c=confusion_matrix(y_test,y_pred);c
