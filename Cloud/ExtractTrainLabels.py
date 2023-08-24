import os
import numpy as np
import json
import pandas as pd


train_folder_name = 'Data/train'
train_label_file = 'Data/class_labels.json'

subdirs_class_labels = os.listdir(train_folder_name)

class_labels_json = {}
subdirs_class_labels = json.dumps(subdirs_class_labels)
subdirs_class_labels = json.loads(subdirs_class_labels)
class_labels_json['train_class_names'] = subdirs_class_labels


with open(train_label_file,'w') as file_obj:
    json.dump(class_labels_json,file_obj,indent=4)




