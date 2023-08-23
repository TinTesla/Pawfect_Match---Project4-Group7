import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import json
import loadconfig


appconfig = loadconfig.Appconfig()
root_directory = os.getcwd()
user_data_directory = 'User_Data'
user_data_path = root_directory + '/' + user_data_directory

#previous models
#model = load_model('models/dog_model_v1.h5')
#model = load_model('models/dog_model_v_1691444284.h5')

#reading the model evaluation details to get the model name to create the validation result file
#model_evaluation_file = 'Data/output/outputModel_evaluation_1691869977.json'
model_evaluation_file = appconfig.evalulation_file
class_names = []
model_name = None
with open(model_evaluation_file,'r') as file_obj:
    eval_data = json.load(file_obj)
    model_name = eval_data.get('model_name')
    model_name = 'models/' + model_name
    class_names = eval_data.get('class_names')


model = load_model(model_name)
#model = load_model('models/dog_model_v_1691779572.h5')
#model = load_model('models/dog_model_v_1690812574.model')

#fetching the training class_labels from the input json in the appconfig file
train_class_labels_file = 'Data/class_labels.json'
with open(train_class_labels_file,'r') as train_class_obj:
    train_class_labels = json.load(train_class_obj)

    species_labels = train_class_labels.get('train_class_names')
    #print("species_labels",species_labels)


class imagePredictor:

    def getPredictedClass(self, image_file):

        img = image.load_img(image_file, target_size=(299, 299))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        #subdirs = os.listdir('Data/train')

        pred = model.predict(img)

        return pred


    def getPredictions(self,image_file):


        img = image.load_img(image_file,target_size=(299,299))
        img = image.img_to_array(img)
        img = img/255.0
        img = np.expand_dims(img,axis=0)

        #subdirs = os.listdir('Data/train')


        pred = model.predict(img)

        predictions = []

        if len(pred) > 0:

            for pr in pred:
                #predicted_species_name = subdirs[pr.argmax()]

                predicted_species_name = species_labels[pr.argmax()]
                per = round(np.amax(pred) * 100, 2)

                if per >= 40:
                    pr_obj = dict(Species = predicted_species_name, Predicted_percent = per)
                    predictions.append(pr_obj)

        return predictions