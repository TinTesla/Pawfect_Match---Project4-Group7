import os
import json
import numpy as np
import pandas as pd
import Predict_Breed as pbr
import loadconfig
from tensorflow.keras.preprocessing import image



Image_Predictor = pbr.imagePredictor()
appconfig = loadconfig.Appconfig()

#for 10 samples of each class
#validation_folder_name = 'Data/validation'
validation_folder_name = 'Data/test'
all_class_labels = os.listdir(validation_folder_name)
all_class_index = [all_class_labels.index(x) for x in all_class_labels]
image_predictions = []



#reading the model evaluation details to get the model name to create the validation result file
#model_evaluation_file = 'Data/output/outputModel_evaluation_1691869977.json'
model_evaluation_file = appconfig.evalulation_file

with open(model_evaluation_file,'r') as file_obj:
    eval_data = json.load(file_obj)
    model_name = eval_data.get('model_name')
    if model_name:
        model_name = model_name.replace('.h5','').replace('.model','').strip()
        validation_result_file = 'Data/output/validation_result_{0}.json'.format(model_name)



        for class_name in all_class_labels:
            class_index = all_class_labels.index(class_name)
            sub_folder_name = validation_folder_name + '/' + class_name
            image_files = os.listdir(sub_folder_name)
            image_files = [sub_folder_name + '/' + x for x in image_files]
            print(image_files)


            for image_file in image_files:
                prediction = Image_Predictor.getPredictions(image_file=image_file)
                if len(prediction) > 0:
                    prediction = prediction[0]
                    predicted_class = prediction.get('Species')
                    prediction_obj = dict(Expected_Class = class_name,Predicted_Class = predicted_class, Image_url = image_file)
                    image_predictions.append(prediction_obj)

            #image_predictions = [Image_Predictor.getPredictedClass(image_file=x) for x in image_files]


        with open(validation_result_file,'w') as file_obj:
            json.dump(image_predictions,file_obj,indent=4)