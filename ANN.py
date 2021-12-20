import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Loading the dataset..!
data = pd.read_csv("C:/Users/******/*******/diabetes.csv")

#Finding How Many Classes There Are..!
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

print("Average Education Loss= ", np.mean(model.history.history["loss"]))

print("Average Educational Performance= ", np.mean(model.history.history["accuracy"]))

print("Average Loss of Validation= ", np.mean(model.history.history["val_loss"]))

print("Average Verification Performance= ", np.mean(model.history.history["val_accuracy"]))


#Achievement Chart


import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Achievement")
plt.ylabel("Achievement")
plt.xlabel("Epoch Number")
plt.legend(["Education", "Test"], loc = "upper left" )
plt.show


#Loss Chart


import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch Number")
plt.legend(["Education", "Test"], loc = "upper left" )
plt.show


import sklearn.metrics as metrics

probs = model.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.title("Accepted")
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Positive Correct Range')
plt.xlabel('Positive False Range')
plt.show()





