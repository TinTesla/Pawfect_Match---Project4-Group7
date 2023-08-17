import streamlit as st
import pandas as pd
import numpy as np
import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


st.image('../sourcedata/dogsload.jpg')

st.write("""
# Simple Dog Breed Match Prediction App

This app predicts the pawfect **Dog Breed ** for you!
""")

st.sidebar.header('User Input Parameters') 

def user_input_features():
    min_height = st.sidebar.slider('Minimum Height', 10, 100, 10)
    max_height = st.sidebar.slider('Maximum Height', 10,100,15)
    min_weight = st.sidebar.slider('Minimum Weight', 0, 79, 10)
    max_weight = st.sidebar.slider('Maximum Weight', 0, 120, 35)
    min_expectancy = st.sidebar.slider('Minimum expactancy', 0, 16, 5)
    max_expectancy = st.sidebar.slider('Maximum expactancy', 0, 19, 11)
    popularity = st.sidebar.slider('How popular your would like your freind to be', 0, 120, 55)
    #groups_kwag = ['Sporting', 'Hound', 'Working', 'Terrier', 'Toy', 'Non-Sporting',  'Herding']
    #group_type = st.sidebar.select_slider('Select ayour type of dog', options=groups_kwag)
    #group = groups_kwag.index(group_type)
    grooming_frequency_value = st.sidebar.slider('grooming_frequency_value', 0.0,1.0,0.2,step=0.20)
    shedding_value = st.sidebar.slider('shedding_value',0.0,1.0,0.2,step=0.20)
    energy_level_value = st.sidebar.slider('energy_level_value', 0.0,1.0, 0.2,step=0.20)
    trainability_value = st.sidebar.slider('trainability_value', 0.0,1.0,0.2,step=0.20)
    demeanor_value = st.sidebar.slider('demeanor_value',  0.0,1.0,0.2,step=0.20)
    data = { 
        'min_height' : min_height, 'max_height' :max_height, 
       'min_weight' : min_weight, 'max_weight' : max_weight, 
       'min_expectancy' : min_expectancy, 'max_expectancy': max_expectancy, 
       'popularity' : popularity,
        'grooming_frequency_value': grooming_frequency_value , 'shedding_value':shedding_value,
       'energy_level_value':energy_level_value, 'trainability_value':trainability_value, 'demeanor_value':demeanor_value        
            }
    
    features = pd.DataFrame(data, index=[0])
    return features

user_input_df = user_input_features()

st.subheader('User Input parameters')
st.write(user_input_df)

dog_breed_val_df = pd.read_csv('../sourcedata/dog_breed_val.csv')
# Get dog breeds as array
dog_breeds = dog_breed_val_df['breed_name'].values
dog_breeds_csv = ', '.join(dog_breeds)
print(dog_breeds_csv)

y = dog_breed_val_df['breed_name']
col_to_drop = ['breed_name','description']
X = dog_breed_val_df.drop(columns=col_to_drop,axis=1) #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
st.write('# Simple Dog Breed Match Prediction App Accuracy', accuracy)


# Generate a classification report for more detailed metrics
# report = classification_report(y_test, y_pred, target_names=dog_breeds)
# print("Classification Report:\n", report)

# # Assuming user_inputs_processed is preprocessed user inputs
# recommended_breed_index = clf.predict(user_input_df)
# recommended_breed_name = dog_breeds[recommended_breed_index[0]]

# print(f"Recommended Dog Breed: {recommended_breed_name}")


