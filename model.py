import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd


# Load the data
df = pd.read_csv('diabetes.csv')

# split the data into training and testing
train, test = train_test_split(df, test_size=0.2)

x_train = []
y_train = []

x_test = []
y_test = []

# format the testing data
for i, j in test.iterrows():
    x_test.append(j[:7].values)
    if j.Outcome == 1:
        y_test.append(np.array([0, 1]))
    else:
        y_test.append(np.array([1,0]))


# format the training data
for i, j in train.iterrows():
    x_train.append(j[:7].values)
    if j.Outcome == 1:
        y_train.append(np.array([0, 1]))
    else:
        y_train.append(np.array([1,0]))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_train = np.array(y_train)


nodes = 20

layers = 3

# initialize sequential model
model = Sequential()
# initialize the numbers using a normal distribution
kernel_initializer = keras.initializers.RandomNormal(mean=0., stddev=30)
bias_initializer = keras.initializers.RandomNormal(mean=0., stddev=10)


model.add(Dense(nodes, input_shape=(7,), activation='sigmoid', name='input',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
for i in range(layers-1):
    model.add(Dense(nodes, activation='sigmoid', name=f'hidden_{i}'))

model.add(Dense(2, activation='softmax', name='output'))
optimizer = Adam(lr=0.0005)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=4000, verbose=0)

results = model.evaluate(x_test, y_test)
print(confusion_matrix(y_test, results))




