# Фильтрация спама
# бинарная классификация
# Векторизация

# Столбцы = слова (в тексте)
# Строки = образцы текста
#  Ячейка = кол-во данных слов в данном тексте

# Очистка: строчные буквы, удаляют знаки препинан, (стоп-слова),

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score


data = pd.read_csv('spam.csv')
print(data.head())

data["Spam"] = data["Category"].apply(lambda  x: 1 if x == "Spam" else 0)

print(data.columns)
print(data.info())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])
w = vectorizer.get_feature_names_out()

# print(w)
# print(w[:, 1000])

X_tr, X_tst, y_tr, y_tst = train_test_split(data["Message"], data["Spam"], test_size=0.25)

md = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())])
md.fit(X_tr, y_tr)

texts = ['Hi! How are you?','Win the lottery',
        'Free subscription', 'Black friday big discount shop offer', 'Nice to meet you']

print(md.predict(texts))

# Фишинг

data = pd.read_csv('phishing.csv')
print(data.head())
print(data.columns)

X = data.drop(columns=["class"])
print(X.columns)
y = pd.DataFrame(data["class"])
print(y.columns)

X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.25)
dt = DecisionTreeClassifier()

model = dt.fit(X_tr, y_tr)

predict = model.predict(X_tst)
print(accuracy_score(predict, y_tst))

# Классификации: бинарные(двоичные), мультиклассовые, многометочные
# - Точность (precision) - стоимость ложных срабатывания высока
# - Полнота (recall) - стоимость ложноотрицательных срабатываний высока
# - Специфичность (specificity) = полнота (наоборот). Насколько точно определяются отрицательные образцы
# - Чувствительность (sensitivity) = полнота
# - F1-мера

# Метрики: - процент ошибок, процент правильных ответов (accuracy)
# Типы ошибок: ложноположительные (ложная тревога), ложноотрицательные (ложный пропуск)
# Типы правильных ответов: истинноположительные, истинноотрицательные


# Аномалии

data = pd.read_csv('creditcard.csv')
print(data.head())

legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]

X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.25)

model1 = LogisticRegression()
model1.fit(X_tr, y_tr)
ConfusionMatrixDisplay.from_estimator(model1, X_tst, y_tst, display_labels=['Легитимная', 'Мошенническая'])
# Точность
y_pred = model1.predict(X_tst)

# Полнота
print(precision_score(y_tst, y_pred))

# Специфичность
print(recall_score(y_tst, y_pred, pos_label=0))




plt.show()



# model2 = RandomForestClassifier(n_estimators=10)
# model2.fit(X_tr, y_tr)
# ConfusionMatrixDisplay.from_estimator(model2, X_tst, y_tst, display_labels=['Легитимная', 'Мошенническая'])
# plt.show()

model3 = GradientBoostingClassifier()
model3.fit(X_tr, y_tr)
ConfusionMatrixDisplay.from_estimator(model3, X_tst, y_tst, display_labels=['Легитимная', 'Мошенническая'])
plt.show()

















