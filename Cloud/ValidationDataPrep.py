import os
import pandas as pd
import json
import shutil
import random
from datetime import datetime

date_format = "%Y-%m-%d %H:%M:%S"

print(datetime.strftime(datetime.now(),date_format))

root_folder = os.getcwd()
train_data_path_src = 'Data/test'
disp_folder_name = 'Data/validation'

all_folders = os.listdir(train_data_path_src)
print(all_folders)

disp_folders_to_create =[]
for folder in all_folders:
    folder_to_create = root_folder + '/' + disp_folder_name + '/' + folder
    disp_folders_to_create.append(folder_to_create)

for folder in disp_folders_to_create:
    if not os.path.exists(folder):
        os.mkdir(folder)


for train_folder in all_folders:
    train_folder_idx = all_folders.index(train_folder)
    display_folder = disp_folders_to_create[train_folder_idx] + '/'
    train_image_folder = train_data_path_src + '/' + train_folder
    random_files = random.sample(os.listdir(train_image_folder),10)
    random_files = [train_data_path_src + '/' + train_folder + '/' + x for x in random_files]
    print(random_files)

    for random_file in random_files:
        shutil.copy(src=random_file,dst=display_folder)

