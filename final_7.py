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
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
    model.add(layers.BatchNormalization())
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


model.save('my_model_7_layers')

model = tf.keras.models.load_model('my_model_7_layers')

# ##Getting Activations
##1

model2 = models.Sequential()

model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model.add(layers.BatchNormalization())

model2.layers[1].set_weights(model.layers[1].get_weights())

layer_1_activations = model2.predict(test_images)
layer_1_activations_mean = layer_1_activations.mean(axis=0)


##2
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[3].set_weights(model.layers[3].get_weights())
model2.add(layers.BatchNormalization())

layer_2_activations = model2.predict(test_images)
layer_2_activations_mean = layer_2_activations.mean(axis=0)


##3
##2
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[3].set_weights(model.layers[3].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[5].set_weights(model.layers[5].get_weights())
model2.add(layers.BatchNormalization())

layer_3_activations = model2.predict(test_images)
layer_3_activations_mean = layer_3_activations.mean(axis=0)




##4
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[3].set_weights(model.layers[3].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[5].set_weights(model.layers[5].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[7].set_weights(model.layers[7].get_weights())
model2.add(layers.BatchNormalization())


layer_4_activations = model2.predict(test_images)
layer_4_activations_mean = layer_4_activations.mean(axis=0)


##5
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[3].set_weights(model.layers[3].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[5].set_weights(model.layers[5].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[7].set_weights(model.layers[7].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[9].set_weights(model.layers[9].get_weights())
model2.add(layers.BatchNormalization())


layer_5_activations = model2.predict(test_images)
layer_5_activations_mean = layer_5_activations.mean(axis=0)


#6
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[3].set_weights(model.layers[3].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[5].set_weights(model.layers[5].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[7].set_weights(model.layers[7].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[9].set_weights(model.layers[9].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[11].set_weights(model.layers[11].get_weights())
model2.add(layers.BatchNormalization())


layer_6_activations = model2.predict(test_images)
layer_6_activations_mean = layer_6_activations.mean(axis=0)


#7
model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(32, 32, 3)))
model2.add(layers.Flatten())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[3].set_weights(model.layers[3].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[5].set_weights(model.layers[5].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[7].set_weights(model.layers[7].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[9].set_weights(model.layers[9].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[11].set_weights(model.layers[11].get_weights())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model2.layers[13].set_weights(model.layers[13].get_weights())
model2.add(layers.BatchNormalization())


layer_7_activations = model2.predict(test_images)
layer_7_activations_mean = layer_7_activations.mean(axis=0)




###Mask
layer_1_mask = np.where(layer_1_activations_mean>(np.mean(layer_1_activations_mean)**2),1,0).reshape((1,100))

layer_2_mask = np.where(layer_2_activations_mean>(np.mean(layer_2_activations_mean)**2),1,0).reshape((1,100))

layer_3_mask = np.where(layer_3_activations_mean>(np.mean(layer_3_activations_mean)**2),1,0).reshape((1,100))

layer_4_mask = np.where(layer_4_activations_mean>(np.mean(layer_4_activations_mean)**2),1,0).reshape((1,100))

layer_5_mask = np.where(layer_5_activations_mean>(np.mean(layer_5_activations_mean)**2),1,0).reshape((1,100))

layer_6_mask = np.where(layer_6_activations_mean>(np.mean(layer_6_activations_mean)**2),1,0).reshape((1,100))

layer_7_mask = np.where(layer_6_activations_mean>(np.mean(layer_7_activations_mean)**2),1,0).reshape((1,100))

##Histograms


plt.hist(layer_1_activations, bins=50)
plt.gca().set(title='Layer 1', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_2_activations, bins=50)
plt.gca().set(title='Layer 2', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_3_activations, bins=50)
plt.gca().set(title='Layer 3', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_4_activations, bins=50)
plt.gca().set(title='Layer 4', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_5_activations, bins=50)
plt.gca().set(title='Layer 5', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_6_activations, bins=50)
plt.gca().set(title='Layer 6', ylabel='Frequency', xlabel='Activation Values');

plt.hist(layer_7_activations, bins=50)
plt.gca().set(title='Layer 7', ylabel='Frequency', xlabel='Activation Values');




###Count above threshold

def get_count_above_threshold(l,t):
    counter = 0
    for i in l:
        if i >= t:
            counter=counter+1
    return counter

steps_for_inc = 0.0001

x1 = np.arange(0,np.max(layer_1_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_1 = []
for i in x1:
    count_below_threshold_1.append(get_count_above_threshold(layer_1_activations_mean, i))
plt.plot(x1,count_below_threshold_1,label='Layer 1')


x2 = np.arange(0,np.max(layer_2_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_2 = []
for i in x2:
    count_below_threshold_2.append(get_count_above_threshold(layer_2_activations_mean, i))
plt.plot(x2,count_below_threshold_2,label='Layer 2')


x3 = np.arange(0,np.max(layer_3_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_3 = []
for i in x3:
    count_below_threshold_3.append(get_count_above_threshold(layer_3_activations_mean, i))
plt.plot(x3,count_below_threshold_3,label='Layer 3')

x4 = np.arange(0,np.max(layer_4_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_4 = []
for i in x4:
    count_below_threshold_4.append(get_count_above_threshold(layer_4_activations_mean, i))
plt.plot(x4,count_below_threshold_4,label='Layer 4')

x5 = np.arange(0,np.max(layer_5_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_5 = []
for i in x5:
    count_below_threshold_5.append(get_count_above_threshold(layer_5_activations_mean, i))
plt.plot(x5,count_below_threshold_5,label='Layer 5')

x6 = np.arange(0,np.max(layer_6_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_6 = []
for i in x6:
    count_below_threshold_6.append(get_count_above_threshold(layer_6_activations_mean, i))
plt.plot(x6,count_below_threshold_6,label='Layer 6')

x7 = np.arange(0,np.max(layer_7_activations)+steps_for_inc,steps_for_inc)
count_below_threshold_7 = []
for i in x7:
    count_below_threshold_7.append(get_count_above_threshold(layer_7_activations_mean, i))
plt.plot(x7,count_below_threshold_7,label='Layer 7')



plt.xlabel('Threshold')
plt.ylabel('Count')
plt.title('Count Above Threshold')
plt.legend()
plt.show()


###Masking
g = 0.5

layer_1_mask = np.where(layer_1_activations_mean>(np.mean(layer_1_activations_mean)**g),1,0).reshape((1,100))

layer_2_mask = np.where(layer_2_activations_mean>(np.mean(layer_2_activations_mean)**g),1,0).reshape((1,100))

layer_3_mask = np.where(layer_3_activations_mean>(np.mean(layer_3_activations_mean)**g),1,0).reshape((1,100))

layer_4_mask = np.where(layer_4_activations_mean>(np.mean(layer_4_activations_mean)**g),1,0).reshape((1,100))

layer_5_mask = np.where(layer_5_activations_mean>(np.mean(layer_5_activations_mean)**g),1,0).reshape((1,100))

layer_6_mask = np.where(layer_6_activations_mean>(np.mean(layer_6_activations_mean)**g),1,0).reshape((1,100))

layer_7_mask = np.where(layer_7_activations_mean>(np.mean(layer_7_activations_mean)**g),1,0).reshape((1,100))



##Fine tuning

model3 = models.Sequential()
model3.add(layers.InputLayer(input_shape=(32, 32, 3)))
model3.add(layers.Flatten())

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_1_mask))

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_2_mask))

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_3_mask))

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_4_mask))

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_5_mask))

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_6_mask))

model3.add(layers.Dense(100, activation='relu',kernel_regularizer='l2'))
model3.add(layers.BatchNormalization())
model3.add(layers.Lambda(lambda x:x*layer_7_mask))


model3.add(layers.Dense(10))
model3.layers[1].set_weights(model.layers[1].get_weights())
model3.layers[4].set_weights(model.layers[3].get_weights())
model3.layers[7].set_weights(model.layers[5].get_weights())
model3.layers[10].set_weights(model.layers[7].get_weights())
model3.layers[13].set_weights(model.layers[9].get_weights())
model3.layers[16].set_weights(model.layers[11].get_weights())
model3.layers[19].set_weights(model.layers[13].get_weights())

model3.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'],run_eagerly=True)




epoch_list =[]
loss_list = []
label_list = []

start_time = time.time()
history2 = model3.fit(train_images, train_labels,batch_size=128,epochs=total_epochs, validation_data=(test_images, test_labels))
end_time = time.time()


 
plt.plot(history.history['val_accuracy'],label='Original')
plt.plot(history2.history['val_accuracy'],label='Pruned')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Performance Comparison')
plt.legend()



def reduction(l):
    count = 0
    for i in l[0]:
        if i == 1:
            count = count+1;
    return count/len(l[0])



reduction_list = []
reduction_list.append(reduction(layer_1_mask))
reduction_list.append(reduction(layer_2_mask))
reduction_list.append(reduction(layer_3_mask))
reduction_list.append(reduction(layer_4_mask))
reduction_list.append(reduction(layer_5_mask))
reduction_list.append(reduction(layer_6_mask))
reduction_list.append(reduction(layer_7_mask))

x_list = [1,2,3,4,5,6,7]
plt.plot(x_list,reduction_list)
plt.ylabel('Pruned/Original')
plt.xlabel('Layer Number')
plt.title('Extent of Pruning - Layerwise')