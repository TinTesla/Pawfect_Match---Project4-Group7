import difflib

import pandas as pd
import sqlite3
import os
from difflib import Differ


root_directory = os.getcwd()
db_folder = 'db'
db_name = 'dogdata.db'
db_path = root_directory + '/' + db_folder + '/' + db_name

sqliteConn = sqlite3.connect(db_path)

akc_data_file_path = 'Data/akc-data-latest.csv'

akc_raw_data_df = pd.read_csv(akc_data_file_path,sep=',')
#akc_raw_data_df.to_sql('dog_species',con= sqliteConn,if_exists='replace',index=False)


akc_raw_data_df = akc_raw_data_df.rename(columns={'Unnamed: 0': 'Dog_Species'})


unique_dog_species = list(akc_raw_data_df['Dog_Species'].drop_duplicates())

class_labels = os.listdir('Data/train')

class_labels = [x.lower() for x in class_labels]
label_mapping = []

for species in unique_dog_species:
    close_matches = difflib.get_close_matches(species.lower(), class_labels)
    if len(close_matches) > 0:
        label_map = dict(Species=species,classlabel=close_matches[0])
        label_mapping.append(label_map)
        print(species, close_matches[0])


df_label_map = pd.DataFrame(label_mapping)
df_label_map.to_sql('Label_Mapping',con= sqliteConn,if_exists='replace',index=False)

    # for cl in class_labels:
    #     difference = difflib.get_close_matches(species.lower(),cl.lower())
    #     print(species,cl,difference)


