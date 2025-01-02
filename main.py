import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_picture(picture):
    plt.imshow(picture)
    plt.show()

def convertToHotOne(y_tr, y_valid, y_tst):
    y_tr = tf.keras.utils.to_categorical(y_tr, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    y_tst = tf.keras.utils.to_categorical(y_tst, 10)
    return y_tr, y_valid, y_tst

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[:5], y_test[:5])
print(convertToHotOne(y_train[:5], y_test[:5]))


X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print('Train images: ', X_train.shape)
print('Train labels: ', y_train.shape)

print('Validation images: ', X_valid.shape)
print('Validation labels: ', y_valid.shape)

print('Test images: ', x_test.shape)
print('Test labels: ', y_test.shape)


