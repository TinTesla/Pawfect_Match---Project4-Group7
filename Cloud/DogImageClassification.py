# USing Pre-Trained Model as Feature Extractor Preprocessor
#The pre-trained model is used as a standalone program to extract features from new photographs.
# using ' Inception V3 ' model for giving features and  building feed forward networks for identifying the breed based on features.

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model,load_model
from glob import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Activation,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import PIL
import time
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt





root_directory = os.getcwd()
model_location = 'models'
model_path = root_directory + '/' + model_location
model_version = str(int(time.time()))
model_name = 'dog_model_v_{0}.model'.format(model_version)
model_file_name = model_path + '/' + model_name
train_folder_name = 'Data/train'
validation_folder_name = 'Data/test'
class_names = os.listdir(train_folder_name)
classes = [class_names.index(x) for x in class_names]

label_file_path = 'Data/labels.csv'

df_labels = pd.read_csv(label_file_path,sep=',')
df_labels.set_index('id',inplace=True)
df_labels = df_labels.to_json(orient='index')
labels_lookup = json.loads(df_labels)

#to get labels from file name
def get_labelIndexfromfilename(filepath):

    filename = Path(filepath).stem
    look_up_val = labels_lookup.get(filename)
    label_index = None
    if look_up_val:
        label_name = look_up_val.get('breed')
        if label_name:
            label_index = class_names.index(label_name)

    return label_index

#loading the model without top layer
inception_model =InceptionV3(include_top=False,input_shape=[299,299,3])

#As the transfer learning models are trained on huge data we need not train it again.

for layer in inception_model.layers:
    layer.trainable = False

# create the base pre-trained model
x = inception_model.output
# add a global spatial average pooling layer
x =GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#predictions = Dense(30, activation='softmax')(x)
x = Dense(120)(x)

pred = Activation('softmax')(x)

#Merging the Dense layer over base model

model = Model(inputs=inception_model.input,outputs=pred)
summary = model.summary()

#Generation of Training and Test data

train_datagen =  ImageDataGenerator(rescale=1./255,
                                   shear_range=0.15,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   height_shift_range=0.1
                                   )
test_datagen = ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_directory('Data/train',
                                              target_size=(299,299),
                                              batch_size=16,
                                              class_mode='categorical'
                                             )

test_set = train_datagen.flow_from_directory('Data/test',
                                              target_size=(299,299),
                                              batch_size=16,
                                              class_mode='categorical'
                                             )

#for confusion matrix - preparing the labels for validation data
test_files = os.listdir('Data/test')

validation_labels = []

for path, subdirs, files in os.walk('Data/test'):
    for name in files:
        filepath = os.path.join(path, name)
        labelIndex = get_labelIndexfromfilename(filepath=filepath)
        validation_labels.append(labelIndex)


print('validation labels')
print(validation_labels)
#Checkpoint - Creating the best model
#Compilation of model with Adam Optimizer and Crossentropy loss

checkpoint = ModelCheckpoint(model_name,monitor='val_accuracy',verbose=1,save_best_only=True)
callback = [checkpoint]

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer= 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

#training the model
#training_history = model.fit(train_set,epochs=2,steps_per_epoch=len(train_set),validation_data=test_set,verbose=1,callbacks=callback)
training_history = model.fit(train_set,epochs=1,steps_per_epoch=2,validation_data=test_set,verbose=1,callbacks=callback)

print(training_history.history)
with open('Data/training_history.json', 'w') as tobj:
    json.dump(training_history.history,tobj,indent=4)

#confusion matrix
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)

print(y_pred.tolist())

print(len(y_pred.tolist()))
#
# cm = confusion_matrix(test_set.classes , y_pred)
# ConfusionMatrixDisplay(cm).plot()
#
# # cm_plot.figure_.savefig('cfm1.png')
#
# # cm_plot.plot()
# plt.savefig('cfm.png')
# confusionMatrix = cm.tolist()
#
# with open('Data/Model_evaluation.json','w') as file_obj:
#     model_eval_json = {}
#     model_eval_json['model_name'] = model_name
#     model_eval_json['class_names'] = class_names
#     model_eval_json['class_index'] = classes
#     model_eval_json['confusionMatrix'] = confusionMatrix
#     model_eval_json['training_history'] = training_history.history
#
#     json.dump(model_eval_json,file_obj,indent=4)
#
#

# con_mat = tf.math.confusion_matrix(labels=validation_labels, predictions=y_pred).numpy()
#
# #Normalization Confusion Matrix to the interpretation of which class is being misclassified.
# con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
#
# con_mat_df = pd.DataFrame(con_mat_norm,
#                           index=classes,
#                           columns=classes)
# con_mat_df = con_mat_df.to_json(orient='records')
# con_mat_df = json.loads(con_mat_df)
#
# with open('Data/confusion_matrix.json', 'w') as tobj:
#     json.dump(con_mat_df,tobj,indent=4)