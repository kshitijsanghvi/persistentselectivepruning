import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import time
total_epochs = 500
X = []
for i in range(1,total_epochs+1):
    X.append(i)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

epoch_list =[]
loss_list = []
label_list = []


tf.config.experimental_run_functions_eagerly(True)

class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(model.layers[1].output.numpy())



optimizer_list = ['adagrad']
time_to_train_list = []
accuracy_list = []
for opt in optimizer_list:
    tf.keras.backend.clear_session()
    model = models.Sequential()
    
    model.add(layers.InputLayer(input_shape=(32, 32, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'],run_eagerly=True)
    
    start_time = time.time()
    history = model.fit(train_images, train_labels,batch_size=128,epochs=total_epochs, validation_data=(test_images, test_labels))
    end_time = time.time()
    time_to_train_list.append(end_time-start_time)
    epoch_list.append(X)
    loss_list.append(history.history['loss'])
    label_list.append(str(opt))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    accuracy_list.append(test_acc)

















    
# for i,loss in enumerate(loss_list):
#     plt.plot(epoch_list[i],loss, label='Original')


# ##Getting Activations
##1

model2 = models.Sequential()

model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))

model2.layers[1].set_weights(model.layers[1].get_weights())

layer_1_activations = model2.predict(test_images)
layer_1_activations_mean = layer_1_activations.mean(axis=0)


##2
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[2].set_weights(model.layers[2].get_weights())


layer_2_activations = model2.predict(test_images)
layer_2_activations_mean = layer_2_activations.mean(axis=0)




layer_1_mask = np.where(layer_1_activations_mean>0.05,1,0).reshape((1,100))

layer_2_mask = np.where(layer_2_activations_mean>0.05,1,0).reshape((1,100))

# layer_1_mask = np.where(layer_1_activations_mean>0.05,1,0)

# layer_2_mask = np.where(layer_2_activations_mean>0.05,1,0)




##3
model3 = models.Sequential()

model3.add(layers.InputLayer(input_shape=(32, 32, 3)))
model3.add(layers.Flatten())
model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2 = models.Sequential()

model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))

model2.layers[1].set_weights(model.layers[1].get_weights())

layer_1_activations = model2.predict(test_images)
layer_1_activations_mean = layer_1_activations.mean(axis=0)


##2
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[2].set_weights(model.layers[2].get_weights())


layer_2_activations = model2.predict(test_images)
layer_2_activations_mean = layer_2_activations.mean(axis=0)



plt.Figure(figsize=(6,6))
plt.plot(history.history['val_accuracy'],label='Original')
plt.plot(history2.history['val_accuracy'],label='Pruned')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Performance Comparison')
plt.legend()
plt.ylim(0,0.6)












plt.hist(layer_1_activations, bins=50)
plt.gca().set(title='Layer 1', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_2_activations, bins=50)
plt.gca().set(title='Layer 2', ylabel='Frequency', xlabel='Activation Values');


def get_count_above_threshold(l,t):
    counter = 0
    for i in l:
        if i >= t:
            counter=counter+1
    return counter

steps_for_inc = 0.01
x1 = np.arange(0,np.max(layer_1_activations)+steps_for_inc,steps_for_inc)

count_below_threshold_1 = []

for i in x1:
    count_below_threshold_1.append(get_count_above_threshold(layer_1_activations_mean, i))


plt.plot(x,count_below_threshold_1,label='Layer 1')


x2 = np.arange(0,np.max(layer_1_activations)+steps_for_inc,steps_for_inc)

count_below_threshold_2 = []

for i in x2:
    count_below_threshold_2.append(get_count_above_threshold(layer_2_activations_mean, i))


plt.plot(x,count_below_threshold_2,label='Layer 2')
plt.xlabel('Threshold')
plt.ylabel('Count')
plt.title('Count Above Threshold')
plt.legend()
plt.show()



##3
layer_1_mask = np.where(layer_1_activations_mean>0.5,1,0).reshape((1,100))

layer_2_mask = np.where(layer_2_activations_mean>0.5,1,0).reshape((1,100))

# layer_1_mask = np.where(layer_1_activations_mean>0.05,1,0)

# layer_2_mask = np.where(layer_2_activations_mean>0.05,1,0)



model3 = models.Sequential()

model3.add(layers.InputLayer(input_shape=(32, 32, 3)))
model3.add(layers.Flatten())
model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.Lambda(lambda x:x*layer_1_mask))
model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.Lambda(lambda x:x*layer_2_mask))

model3.add(layers.Dense(10))
model3.layers[1].set_weights(model.layers[1].get_weights())
model3.layers[3].set_weights(model.layers[2].get_weights())
model3.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'],run_eagerly=True)
 

epoch_list =[]
loss_list = []
label_list = []

start_time = time.time()
history2 = model3.fit(train_images, train_labels,batch_size=128,epochs=total_epochs, validation_data=(test_images, test_labels))
end_time = time.time()
time_to_train_list.append(end_time-start_time)
epoch_list.append(X)
loss_list.append(history.history['loss'])
label_list.append(str(opt))
test_loss, test_acc = model3.evaluate(test_images,  test_labels, verbose=2)
accuracy_list.append(test_acc)
model3.add(layers.Lambda(lambda x:x*layer_1_mask))
model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.Lambda(lambda x:x*layer_2_mask))

model3.add(layers.Dense(10))
model3.layers[1].set_weights(model.layers[1].get_weights())
model3.layers[3].set_weights(model.layers[2].get_weights())
model3.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'],run_eagerly=True)
 

epoch_list =[]
loss_list = []
label_list = []

start_time = time.time()
history2 = model3.fit(train_images, train_labels,batch_size=128,epochs=total_epochs, validation_data=(test_images, test_labels))
end_time = time.time()
time_to_train_list.append(end_time-start_time)
epoch_list.append(X)
loss_list.append(history.history['loss'])
label_list.append(str(opt))
test_loss, test_acc = model3.evaluate(test_images,  test_labels, verbose=2)
accuracy_list.append(test_acc)

# for i,loss in enumerate(loss_list):
#     plt.plot(epoch_list[i],loss, label='Pruning + Fine Tuning')

# plt.grid(True)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
# plt.title('Activation Comparison')
# plt.show()

plt.Figure(figsize=(6,6))
plt.plot(history.history['val_accuracy'],label='Original')
plt.plot(history2.history['val_accuracy'],label='Pruned')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Performance Comparison')
plt.legend()
plt.ylim(0,0.6)




