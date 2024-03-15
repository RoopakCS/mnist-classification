# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.

![minst](https://github.com/RoopakCS/mnist-classification/assets/139228922/e091dc01-8602-4877-8044-9e6edeef758a)


## Neural Network Model

![image](https://github.com/RoopakCS/mnist-classification/assets/139228922/3d1189ef-7b7c-4c9e-908c-0a1fc579fb59)

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict


## PROGRAM

### Name: Roopak C S
### Register Number: 212223220088
## Importing Modules
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
## Loading the MNIST dataset
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
## To view the dimensions of the training and testing dataset
```python
X_train.shape
X_test.shape
```
## Extracting the first image from the training dataset
```python
single_image= X_train[0]
```
## To view the dimensions of the first image from the training dataset
```python
single_image.shape
```
## To visualize the single image 
```python
plt.imshow(single_image,cmap='gray')
```
## To view the dimensions of the labels associated with the training dataset
```python
y_train.shape
```
## To find the minimum and maximum pixel value in the entire training dataset
```python
X_train.min()
X_train.max()
```
## Scaling the pixel values of the images in the MNIST dataset
```python
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
## To find the minimum and maximum pixel value in the entire training dataset
```python
X_train_scaled.min()
X_train_scaled.max()
```
## Extracting the first image from the training dataset
```python
y_train[0]
```
## Using one-hot encoding to convert the target labels into categorical variables
```python
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
## To determine the type of "y_train_onehot"
```python
type(y_train_onehot)
```
##  To view the dimensions of the array
```python
y_train_onehot.shape
```
## Extracting a single image at index 500 from the training dataset
```python
single_image = X_train[500]
```
## To display the image stored in the variable single_image
```python
plt.imshow(single_image,cmap='gray')
```
## Acessing the one-hot encoded representation of the label for the 500th sample in the training dataset
```python
y_train_onehot[500]
```
## Reshaping your training and test data arrays 
```python
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
## Creating the model:
```python
model = Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
```
## To display the summary of the neural network model
```python
model.summary()
```
## Choosing the appropriate parameters
```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
```
## Fitting the model
```python
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
```
## Creating a pandas DataFrame from the training history of the neural network model
```python
metrics = pd.DataFrame(model.history.history)
```
## To display the first five rows of the DataFrame
```
metrics.head()
```
## To visualize the training and validation accuracy
```python
metrics[['accuracy','val_accuracy']].plot()
```
## To visualize the training and validation loss 
```python
metrics[['loss','val_loss']].plot()
```
## To calculate the predictions on the test set using your trained neural network model
```python
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
```
## To print the confusion matrix
```python
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
##Prediction for a single input
```python
img = image.load_img('minst.png')
type(img)
img = image.load_img('minst.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![download (2)](https://github.com/RoopakCS/mnist-classification/assets/139228922/8c745ce8-9858-4926-ba76-5e40086d3272)

![download (3)](https://github.com/RoopakCS/mnist-classification/assets/139228922/1c47f708-f0c4-476c-9866-87cbc838b2c1)

### Classification Report

![image](https://github.com/RoopakCS/mnist-classification/assets/139228922/b785c716-a544-42ac-9339-2e3d37628e51)

### Confusion Matrix

![image](https://github.com/RoopakCS/mnist-classification/assets/139228922/d5e24d65-5c57-4d2c-8b1a-3c4dbed8d493)

### New Sample Data Prediction

![image](https://github.com/RoopakCS/mnist-classification/assets/139228922/45ddd417-dd1e-43b2-9de1-d51bce23a462)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
