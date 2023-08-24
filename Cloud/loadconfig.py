import json
import os

root_directory = os.getcwd()
print(root_directory)
config_file_path = root_directory + '/' + "config/appconfig.json"
class Appconfig:
    #config_file_path = "config/appconfig.json"

    with open(config_file_path,'r') as file_obj:
        config_data = json.load(file_obj)
        evalulation_file = config_data.get("evaluation_file")
        other_evaluation_file = config_data.get("other_evaluation_file")

