#Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not.
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('https://github.com/Devika0901/codsoft/blob/main/Titanic-Dataset.csv')
df.head()
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df.drop(columns=['PassengerId','Name','Ticket','Fare','Cabin'], inplace=True)
df.isnull().sum()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.isnull().mean()
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
sns.countplot(x='Survived', hue='Embarked', data=df)
plt.title('Survival Count by Embarked')
plt.xlabel('Survived')
plt.ylabel('Count')
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.corr()
x = df.drop('Survived', axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.transform(x_test)
def model_train_test(model):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(classification_report(y_test,y_pred))

model_train_test(LogisticRegression())
