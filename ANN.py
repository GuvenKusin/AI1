import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print()
#Veri Seti Yüklenmeli
data = pd.read_csv("C:/Users/bilin/Desktop/diyabet.csv")

#Kaç Sınıf Olduğunu Bulma..!
le = LabelEncoder().fit(data.output)
liste = le.transform(data.output)
classes = list(le.classes_)

x = data.drop(["output"] , axis = 1)
y = liste

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = std.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train) 
categorizated_y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=8, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics= ["accuracy"])

model.fit(x_train, y_train, validation_data=(x_test, categorizated_y_test), epochs=100)

print("Ortalama Eğitim Kaybı= ", np.mean(model.history.history["loss"]))

print("Ortalama Eğitim Başarımı= ", np.mean(model.history.history["accuracy"]))

print("Ortalama Doğrulama Kaybı= ", np.mean(model.history.history["val_loss"]))

print("Ortalama Doğrulama Başarımı= ", np.mean(model.history.history["val_accuracy"]))


#Başarım Grafiği


import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc = "upper left" )
plt.show


#Kayıp Grafiği


import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model kaybı")
plt.ylabel("kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc = "upper left" )
plt.show


import sklearn.metrics as metrics

probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.title("Kabul Edilen")
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('olumlu Doğru Aralık')
plt.xlabel('Olumlu Yanlış Aralık')
plt.show()






