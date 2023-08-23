#dependencies
import streamlit as st
import pandas as pd
import numpy as np
import os
import streamlit as st
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

#
# # setting path for model file
# root_directory = os.getcwd()
# model_name_rf_group = 'dog_app_rf_group_pred.sav'
# model_rf_group_filename = 'models/' + model_name_rf_group
# model_name_rf_breed = 'dog_app_rf_breed_pred.sav'
# model_rf_breed_filename = 'models/' + model_name_rf_breed
#
# #loading models
# loaded_model_rf = pickle.load(open(model_rf_group_filename, 'rb'))
# loaded_model_breed_pred = pickle.load(open(model_rf_breed_filename, 'rb'))
#

# image display
#st.image('../sourcedata/dogsload.jpg')
st.image('image/user_questionaire_app_image2.jpg')

# Headers on page
st.sidebar.header('Find Your Match: Breed Group 📈')
st.sidebar.header('Find Your Match: Breed 🐕')
st.subheader("Dog Breed and Group Match Prediction App")
st.markdown("##### This app predicts the pawfect Dog Match for you! :dog: ")

#@st.cache_data(experimental_allow_widgets=True)

# **************************************get widget values********************************************
def get_group_features():
    # for group fetch
    height_group = st.session_state.get("height_group")
    min_height_group = height_group[0]
    max_height_group = height_group[1]
    weight_group = st.session_state.get("weight_group")
    min_weight_group = weight_group[0]
    max_weight_group = weight_group[1]
    expectancy_group = st.session_state.get("expectancy_group")
    min_expectancy_group = expectancy_group[0]
    max_expectancy_group = expectancy_group[1]
    popularity_group = st.session_state.get("popularity_group")
    grooming_frequency_value_group = st.session_state.get("grooming_group")
    shedding_value_group = st.session_state.get("shedding_group")
    energy_level_value_group = st.session_state.get("energy_group")
    trainability_value_group = st.session_state.get("trainability_group")
    demeanor_value_group = st.session_state.get("demeanor_group")

    group_data = {
        'popularity': popularity_group,'min_height': min_height_group, 'max_height': max_height_group,
        'min_weight': min_weight_group, 'max_weight': max_weight_group,
        'min_expectancy': min_expectancy_group, 'max_expectancy': max_expectancy_group,
        'grooming_frequency_value': grooming_frequency_value_group, 'shedding_value': shedding_value_group,
        'energy_level_value': energy_level_value_group, 'trainability_value': trainability_value_group,
        'demeanor_value': demeanor_value_group
    }
    group_features = pd.DataFrame(group_data, index=[0])

    return group_features

def get_breed_features():
    # for breed fetch
    height_breed = st.session_state.get("height_breed")
    min_height_breed = height_breed[0]
    max_height_breed = height_breed[1]
    weight_breed = st.session_state.get("weight_breed")
    min_weight_breed = weight_breed[0]
    max_weight_breed = weight_breed[1]
    expectancy_breed = st.session_state.get("expectancy_breed")
    min_expectancy_breed = expectancy_breed[0]
    max_expectancy_breed = expectancy_breed[1]
    popularity_breed = st.session_state.get("popularity_breed")
    grooming_frequency_value_breed = st.session_state.get("grooming_breed")
    shedding_value_breed = st.session_state.get("shedding_breed")
    energy_level_value_breed = st.session_state.get("energy_breed")
    trainability_value_breed = st.session_state.get("trainability_breed")
    demeanor_value_breed = st.session_state.get("demeanor_breed")
    groups_kwag = ['Sporting', 'Hound', 'Working', 'Terrier', 'Toy', 'Non-Sporting', 'Herding']
    group_type = st.session_state.get("group_type")
    group = groups_kwag.index(group_type)

    breed_data = {
        'popularity': popularity_breed,'min_height': min_height_breed, 'max_height': max_height_breed,
        'min_weight': min_weight_breed, 'max_weight': max_weight_breed,
        'min_expectancy': min_expectancy_breed, 'max_expectancy': max_expectancy_breed,
        'group': group,'grooming_frequency_value': grooming_frequency_value_breed, 'shedding_value': shedding_value_breed,
        'energy_level_value': energy_level_value_breed, 'trainability_value': trainability_value_breed,
        'demeanor_value': demeanor_value_breed
    }

    breed_features = pd.DataFrame(breed_data, index=[0])

    return breed_features


# *****************creating the slider inputs for Group and Breed prediction in the page in two separate tabs
def user_input_features_group():

    t1,t2 = st.tabs(["###### Predict Group  ","###### Predict Breed"])


    with t1:
        # ******************************For Group Prediction********************************

        st.slider('Minimum, Maximum Height', 10.0, 100.0, (20.0, 80.0), key="height_group")
        st.slider('Minimum Weight, Maximum', 0.0, 79.0, (3.0,30.0),key="weight_group")
        st.slider('Minimum, Maximum expectancy', 0.0, 16.0, (3.0,12.0),key="expectancy_group")
        st.slider('How popular your would like your friend to be', 0, 148, 75,key="popularity_group")
        st.slider('grooming_frequency_value', 0.0,1.0,0.2,step=0.20,key="grooming_group")
        st.slider('shedding_value',0.0,1.0,0.2,step=0.20,key="shedding_group")
        st.slider('energy_level_value', 0.0,1.0, 0.6,step=0.20,key="energy_group")
        st.slider('trainability_value', 0.0,1.0,0.4,step=0.20,key="trainability_group")
        st.slider('demeanor_value',  0.0,1.0,0.2,step=0.20,key="demeanor_group")
        user_input_group = get_group_features()

        # for group prediction based on inputs
        # Remove the index column from the DataFrame
        user_input_no_index = user_input_group.reset_index(drop=True)

        st.divider()

        # Display the DataFrame in Streamlit
        st.subheader("Your input")
        st.write(user_input_no_index.to_html(index=False, escape=False), unsafe_allow_html=True)

        # call prediction function
        #pred_group = group_prediction(user_input_df=user_input_group)
        st.markdown("                                       ")
        # display results
        st.markdown(
            "**Open Colab Notebook [Predict Group](https://colab.research.google.com/drive/1u3M6Fi-xdaPNQYjz4hyJDxhOD4sHNTts)**")


    with t2:
        # ******************************For Breed Prediction********************************

        st.slider('Minimum, Maximum Height', 10.0, 100.0, (20.0, 80.0), key="height_breed")
        st.slider('Minimum Weight, Maximum', 0.0, 79.0, (3.0, 30.0), key="weight_breed")
        st.slider('Minimum, Maximum expectancy', 0.0, 16.0, (3.0, 12.0), key="expectancy_breed")
        st.slider('How popular your would like your friend to be', 0, 148, 75, key="popularity_breed")
        st.slider('grooming_frequency_value', 0.0, 1.0, 0.2, step=0.20, key="grooming_breed")
        st.slider('shedding_value', 0.0, 1.0, 0.2, step=0.20, key="shedding_breed")
        st.slider('energy_level_value', 0.0, 1.0, 0.6, step=0.20, key="energy_breed")
        st.slider('trainability_value', 0.0, 1.0, 0.4, step=0.20, key="trainability_breed")
        st.slider('demeanor_value', 0.0, 1.0, 0.2, step=0.20, key="demeanor_breed")

        # for breed prediction based on group
        groups_kwag = ['Sporting', 'Hound', 'Working', 'Terrier', 'Toy', 'Non-Sporting', 'Herding']
        st.select_slider('Select ayour type of dog', options=groups_kwag,key="group_type")

        user_input_breed = get_breed_features()

        # for breed prediction
        # Remove the index column from the DataFrame
        breed_user_input_no_index = user_input_breed.reset_index(drop=True)

        st.divider()

        # Display the DataFrame in Streamlit
        st.subheader("Your input")
        st.write(breed_user_input_no_index.to_html(index=False, escape=False), unsafe_allow_html=True)

        # prediction function call
        #pred_breed = breed_prediction(user_input_df=user_input_breed)

        st.markdown("                                       ")
        # display results
        st.markdown(
            "**Open Colab Notebook [Predict Breed](https://colab.research.google.com/drive/1ADi6US2i3o-M-hhf8JaibRzlJRzuYsNI?usp=sharing)**")


# ***************************************************************************************************************
# main prediction function call
# reading the csv for mapping, group mapping and calling the function to fetch user inputs from slider for both
# group prediction and breed prediction
# ***************************************************************************************************************
dog_map_group_df = pd.read_csv('Data/output/dog_group_mapping.csv')
# reading the csv for mapping, breed mapping
dog_map_breed_df = pd.read_csv('Data/output/dog_breed_mapping.csv')

# Group Prediction based on Featureset
# def group_prediction(user_input_df):
#
#     # load the model from disk
#     #filename = 'dog_app_rf_group_pred.sav'
#     input_array = user_input_df.values
#
#     input_array_reshaped = input_array.reshape(1, -1)  # Reshape to match the model's input shape
#
#     # Make predictions
#     predicted_group = loaded_model_rf.predict(input_array_reshaped)[0]
#     # print("Predicted Group: " + str(predicted_group))
#     # dog_map_group_df = pd.read_csv('Data/output/dog_group_mapping.csv')
#     # result = dog_map_group_df['group'].iloc[predicted_group]
#     # print(result)
#     return predicted_group

# Breed Prediction Function
# def breed_prediction(user_input_df):
#
#     # load the model from disk
#     #filename = 'dog_app_rf_breed_pred.sav'
#
#     input_array = user_input_df.values
#
#     input_array_reshaped = input_array.reshape(1, -1)  # Reshape to match the model's input shape
#
#     # Make predictions
#     predicted_class = loaded_model_breed_pred.predict(input_array_reshaped)[0]
#
#     # Display results
#
#     # dog_map_breed_df = pd.read_csv('../sourcedata/dog_breed_mapping.csv')
#     # st.write("Predicted Class:", dog_map_breed_df.iloc[predicted_class])
#
#     return predicted_class

# Taking the user inputs via sliders
user_input_features_group()

