import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


st.image('../Images/dogsload.jpg')

st.write("""
# Simple Dog Breed Match Prediction App

This app predicts the pawfect **Dog Breed ** for you!
""")

st.sidebar.header('User Input Parameters') 

def user_input_features_group():
    height = st.sidebar.slider('Minimum, Maximum Height', 10.0, 100.0, (10.0,50.0))
    min_height = height[0]
    max_height = height[1]

    weight = st.sidebar.slider('Minimum Weight, Maximum', 0.0, 79.0, (5.0,10.0))
    min_weight = weight[0]
    max_weight = weight[1]

    expectancy = st.sidebar.slider('Minimum, Maximum expactancy', 0.0, 16.0, (3.0,10.0))
    min_expectancy = expectancy[0]
    max_expectancy = expectancy[1]

    popularity = st.sidebar.slider('How popular your would like your freind to be', 0, 148, 55)
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
    
    group_features = pd.DataFrame(data , index=[0])
    
    return group_features

def user_input_features_breed():
    height = st.sidebar.slider('Minimum, Maximum Height', 10.0, 100.0, (10.0,50.0))
    min_height = height[0]
    max_height = height[1]

    weight = st.sidebar.slider('Minimum Weight, Maximum', 0.0, 79.0, (5.0,10.0))
    min_weight = weight[0]
    max_weight = weight[1]

    expectancy = st.sidebar.slider('Minimum, Maximum expactancy', 0.0, 16.0, (3.0,10.0))
    min_expectancy = expectancy[0]
    max_expectancy = expectancy[1]

    groups_kwag = ['Sporting', 'Hound', 'Working', 'Terrier', 'Toy', 'Non-Sporting',  'Herding']
    group_type = st.sidebar.select_slider('Select ayour type of dog', options=groups_kwag)
    group = groups_kwag.index(group_type)

    popularity = st.sidebar.slider('How popular your would like your freind to be', 0, 148, 55)
    grooming_frequency_value = st.sidebar.slider('grooming_frequency_value', 0.0,1.0,0.2,step=0.20)
    shedding_value = st.sidebar.slider('shedding_value',0.0,1.0,0.2,step=0.20)
    energy_level_value = st.sidebar.slider('energy_level_value', 0.0,1.0, 0.2,step=0.20)
    trainability_value = st.sidebar.slider('trainability_value', 0.0,1.0,0.2,step=0.20)
    demeanor_value = st.sidebar.slider('demeanor_value',  0.0,1.0,0.2,step=0.20)
    
    data = { 
        'min_height' : min_height, 'max_height' :max_height, 
       'min_weight' : min_weight, 'max_weight' : max_weight, 
       'min_expectancy' : min_expectancy, 'max_expectancy': max_expectancy, 
       'group' : group,
       'popularity' : popularity,
        'grooming_frequency_value': grooming_frequency_value , 'shedding_value':shedding_value,
       'energy_level_value':energy_level_value, 'trainability_value':trainability_value, 'demeanor_value':demeanor_value        
            }
    
    breed_features = pd.DataFrame(data , index=[0])
    return breed_features


# Group Prediction based on Featureset
def group_prediction():
    st.subheader('User Input parameters')
    user_input_df = user_input_features_group()

    # Remove the index column from the DataFrame
    user_input_no_index = user_input_df.reset_index(drop=True)

    # Display the DataFrame in Streamlit
    st.write(user_input_no_index.to_html(index=False, escape=False), unsafe_allow_html=True)

    # load the model from disk
    filename = 'dog_app_rf_group_pred.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    input_array = user_input_df.values
    scaler = StandardScaler()
    X_scaler = scaler.fit(input_array)
    input_array = input_array.reshape(1, -1)  # Reshape to match the model's input shape

    # Make predictions
    predicted_class = loaded_model.predict(input_array)[0]

    return 

# Display results

dog_map_group_df = pd.read_csv('../sourcedata/dog_group_mapping.csv')
st.write("Predicted Class:", dog_map_group_df.iloc[predicted_class])


# Breed Prediction Function

def breed_prediction():
    breed_user_input = user_input_features_breed()

    # Remove the index column from the DataFrame
    breed_user_input_no_index = breed_user_input.reset_index(drop=True)

    # Display the DataFrame in Streamlit
    st.write(breed_user_input_no_index.to_html(index=False, escape=False), unsafe_allow_html=True)

    # load the model from disk
    filename = 'dog_app_rf_breed_pred.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    input_array = breed_user_input.values
    scaler = StandardScaler()
    X_scaler = scaler.fit(input_array)
    input_array = input_array.reshape(1, -1)  # Reshape to match the model's input shape

    # Make predictions
    predicted_class = loaded_model.predict(input_array)[0]

    # Display results

    dog_map_breed_df = pd.read_csv('../sourcedata/dog_breed_mapping.csv')
    st.write("Predicted Class:", dog_map_breed_df.iloc[predicted_class])

    return




