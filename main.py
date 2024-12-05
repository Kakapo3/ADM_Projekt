import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_picture(picture):
    plt.imshow(picture)
    plt.show()

def convertToHotOne(y_tr, y_tst):
    y_tr = tf.keras.utils.to_categorical(y_tr, 10)
    y_tst = tf.keras.utils.to_categorical(y_tst, 10)
    return y_tst, y_tr

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[:5], y_test[:5])
print(convertToHotOne(y_train[:5], y_test[:5]))



