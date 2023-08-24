import os
import pandas as pd
import json
import shutil
from sklearn.model_selection import train_test_split

root_folder = os.getcwd()
train_data_path = 'Data/train'
test_data_path = 'Data/test'
train_data_path_src = 'Data/train_1'
label_file_path = 'Data/labels.csv'

df = pd.read_csv(label_file_path,sep=',')
df.set_index('id',inplace=True)
df = df.to_json(orient='index')
df_lookup = json.loads(df)

#Step 1
all_files = os.listdir(train_data_path_src)
train_files,test_files = train_test_split(all_files,train_size=0.75, random_state=0,shuffle=True)

train_folders = [(df_lookup.get(os.path.splitext(filename)[0])['breed'],filename)
                  for filename in train_files if df_lookup.get(os.path.splitext(filename)[0])]
print(train_folders)
folders_to_create =[]
for tr in train_folders:
    if not os.path.exists(tr[0]):
        folder_name = tr[0]
        folder_name = root_folder + '/' + train_data_path + '/' + folder_name
        #print( 'creating' + folder_name)
        folders_to_create.append(folder_name)
    #print(tr[1])

folders_to_create = list(set(folders_to_create))

for folder in folders_to_create:
    os.mkdir(folder)

for tr in train_folders:
    src_file_name = root_folder + '/' + train_data_path_src + '/' + tr[1]
    dest_folder = root_folder + '/' + train_data_path + '/' + tr[0] + '/' + tr[1]
    print("Copying file " + src_file_name + 'to ' +  dest_folder)
    shutil.copy(src_file_name,dest_folder)

#**********************For TEST FILE PREP*************************************************************


test_folders = [(df_lookup.get(os.path.splitext(filename)[0])['breed'],filename)
                  for filename in test_files if df_lookup.get(os.path.splitext(filename)[0])]

folders_to_create =[]
for tr in test_folders:
    if not os.path.exists(tr[0]):
        folder_name = tr[0]
        folder_name = root_folder + '/' + test_data_path + '/' + folder_name
        #print( 'creating' + folder_name)
        folders_to_create.append(folder_name)
    #print(tr[1])

folders_to_create = list(set(folders_to_create))

for folder in folders_to_create:
    os.mkdir(folder)

for tr in test_folders:
    src_file_name = root_folder + '/' + train_data_path_src + '/' + tr[1]
    dest_folder = root_folder + '/' + test_data_path + '/' + tr[0] + '/' + tr[1]
    print("Copying file " + src_file_name + 'to ' +  dest_folder)
    shutil.copy(src_file_name,dest_folder)