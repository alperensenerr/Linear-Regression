import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

veriler = pd.read_csv("satislar.csv")
aylar = veriler[["Aylar"]]
satislar = veriler[["Satislar"]]
              
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.2, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
x_test = x_test.sort_index()
y_test = y_test.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.show()
