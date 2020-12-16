import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import matplotlib.pyplot as plt
import time

total_epochs = 200
X = []
for i in range(1,total_epochs+1):
    X.append(i)
    
class CustomCallback(keras.callbacks.Callback):        
    def on_train_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
            print("LOLOLOL")
    
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
optimizer_list = ['adagrad','rmsprop','adadelta','adam','nadam']
time_to_train_list = []
accuracy_list = []
loss_list = []

opt = 'adagrad'

tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(32, 32, 3)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu',kernel_regularizer='l2'))
model.add(layers.Dense(1000, activation='relu',kernel_regularizer='l2'))
model.add(layers.Dense(10))
model.compile(optimizer=opt,
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
start_time = time.time()
history = model.fit(train_images, train_labels,batch_size=128,epochs=total_epochs,validation_data=(test_images, test_labels),callbacks=[CustomCallback()],)
end_time = time.time()
time_to_train_list.append(end_time-start_time)
loss_list.append(history.history['loss'])
test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)
accuracy_list.append(test_acc)

