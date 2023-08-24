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

st.header('Find Your Match by Group 📈')
st.sidebar.header('Find Your Match by Group 📈')

# setting path for model file
root_directory = os.getcwd()
model_rf_group_filename = 'models/dog_app_rf_group_pred'
model_rf_breed_filename = 'models/dog_app_rf_breed_pred'

st.image('image/user_questionaire_app_image2.jpg')
st.subheader("Dog Breed and Group Match Prediction App")
st.markdown("##### This app predicts the pawfect Dog Match for you! :dog: ")

st.header('User Input Parameters') 
# @st.cache_data(experimental_allow_widgets=True)

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
        'min_height': min_height_group, 'max_height': max_height_group,
        'min_weight': min_weight_group, 'max_weight': max_weight_group,
        'min_expectancy': min_expectancy_group, 'max_expectancy': max_expectancy_group,
        'popularity': popularity_group,
        'grooming_frequency_value': grooming_frequency_value_group, 'shedding_value': shedding_value_group,
        'energy_level_value': energy_level_value_group, 'trainability_value': trainability_value_group,
        'demeanor_value': demeanor_value_group
    }
    group_features = pd.DataFrame(group_data, index=[0], columns=list(group_data.keys()))

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
    # groups_kwag = ['Foundation Stock Service', 'Terrier', 'Sporting', 'Working', 'Herding', 'Hound', 'Toy', 'Non-Sporting', 'Miscellaneous']
    group_type = st.session_state.get("group_type")
    # group = groups_kwag.index(group_type)

    breed_data = {
        'min_height': min_height_breed, 'max_height': max_height_breed,
        'min_weight': min_weight_breed, 'max_weight': max_weight_breed,
        'min_expectancy': min_expectancy_breed, 'max_expectancy': max_expectancy_breed,
        'popularity': popularity_breed,
        'grooming_frequency_value': grooming_frequency_value_breed, 'shedding_value': shedding_value_breed,
        'energy_level_value': energy_level_value_breed, 'trainability_value': trainability_value_breed,
        'demeanor_value': demeanor_value_breed,
        'Foundation Stock Service': 0, 'Herding': 0, 'Hound': 0, 'Miscellaneous': 0,
        'Non-Sporting': 0, 'Sporting': 0, 'Terrier': 0, 'Toy': 0, 'Working': 0
    }

    if group_type == 'Sporting':
        breed_data['Sporting'] = 1
    elif group_type == 'Hound':
        breed_data['Hound'] = 1
    elif group_type == 'Working':
        breed_data['Working'] = 1
    elif group_type == 'Terrier':
        breed_data['Terrier'] = 1
    elif group_type == 'Toy':
        breed_data['Toy'] = 1
    elif group_type == 'Non-Sporting':
        breed_data['Non-Sporting'] = 1
    elif group_type == 'Foundation Stock Service':
        breed_data['Foundation Stock Service'] = 1
    elif group_type == 'Miscellaneous':
        breed_data['Miscellaneous'] = 1
    else:
        breed_data['Herding'] = 1

    breed_features = pd.DataFrame(breed_data, index=[0], columns=list(breed_data.keys()))

    return breed_features


# Group Prediction based on Featureset
def group_prediction(user_input_df):

    # load the model from disk
    # filename = 'dog_app_rf_group_pred.sav'
    # loaded_model_rf = pickle.load(open(model_rf_group_filename, 'rb'))
    print(os.listdir('models'))
    with open(model_rf_group_filename, 'rb') as file:
        loaded_model_rf = pickle.load(file)

    input_array = user_input_df.values.reshape(1, -1) # Reshape to match the model's input shape

    # Make predictions
    predicted_group = loaded_model_rf.predict(input_array)
    # print("Predicted Group: " + str(predicted_group))
    dog_map_group_df = pd.read_csv('Data/output/dog_group_mapping.csv')
    result = dog_map_group_df['group'].iloc[predicted_group]
    print(result)
    return predicted_group

# Breed Prediction Function
def breed_prediction(user_input_df):

    # load the model from disk
    #filename = 'dog_app_rf_breed_pred.sav'
    with open(model_rf_breed_filename, 'rb') as file:
        loaded_model_breed_pred = pickle.load(file)

    # loaded_model_breed_pred = pickle.load(open(model_rf_breed_filename, 'rb'))

    input_array = user_input_df.values.reshape(1, -1) # Reshape to match the model's input shape

    # Make predictions
    predicted_class = loaded_model_breed_pred.predict(input_array)

    # Display results
    dog_map_breed_df = pd.read_csv('Data/output/dog_breed_mapping.csv')
    result = dog_map_breed_df['breed'].iloc[predicted_class]
    print(result)
    # st.write("Predicted Class:", dog_map_breed_df.iloc[predicted_class])
    return predicted_class

# read in data for mapping
dog_map_group_df = pd.read_csv('Data/output/dog_group_mapping.csv')
dog_map_breed_df = pd.read_csv('Data/output/dog_breed_mapping.csv')

t1,t2 = st.tabs(["###### Predict Group  ","###### Predict Breed"])


with t1:
    # ******************************For Group Prediction********************************

    st.slider('Minimum, Maximum Height', 5, 100, (20, 80),key="height_group")
    st.slider('Minimum, Maximum Weight', 0, 200, (3, 30),key="weight_group")
    st.slider('Minimum, Maximum expectancy', 0, 17, (3, 12),key="expectancy_group")
    st.slider('How popular your would like your friend to be', 0, 190, 75,key="popularity_group")
    st.slider('Grooming Frequency Value', 0.0,1.0,0.0,step=0.20,key="grooming_group")
    st.slider('Shedding Value',0.0,1.0,0.0,step=0.20,key="shedding_group")
    st.slider('Energy Level Value', 0.0,1.0, 0.0,step=0.20,key="energy_group")
    st.slider('Trainability Value', 0.0,1.0,0.0,step=0.20,key="trainability_group")
    st.slider('Demeanor Value',  0.0,1.0,0.0,step=0.20,key="demeanor_group")

    if st.button('See the Result!', key='submit_1'):

        user_input_group = get_group_features()

        # for group prediction based on inputs
        # Remove the index column from the DataFrame
        user_input_no_index = user_input_group.reset_index(drop=True)

        st.divider()

    # Display the DataFrame in Streamlit
        st.subheader("Your input")
        st.dataframe(user_input_no_index, hide_index=True)
    # st.write(user_input_no_index.to_html(index=False, escape=False), unsafe_allow_html=True)
        predicted_group = group_prediction(user_input_no_index)
    # call prediction function
        # pred_group = group_prediction(user_input_no_index)
        st.markdown("                                       ")
    # display results
        st.write("##### Predicted Group:", dog_map_group_df['group'].iloc[predicted_group])

with t2:
        # ******************************For Breed Prediction********************************

    st.slider('Minimum, Maximum Height', 5, 100, (20, 80), key="height_breed")
    st.slider('Minimum, Maximum Weight', 0, 200, (3, 30), key="weight_breed")
    st.slider('Minimum, Maximum expectancy', 0, 17, (3, 12), key="expectancy_breed")
    st.slider('How popular your would like your friend to be', 0, 190, 75, key="popularity_breed")
    st.slider('Grooming Frequency Value', 0.0, 1.0, 0.0, step=0.20, key="grooming_breed")
    st.slider('Shedding Value', 0.0, 1.0, 0.0, step=0.20, key="shedding_breed")
    st.slider('Energy Level Value', 0.0, 1.0, 0.0, step=0.20, key="energy_breed")
    st.slider('Trainability Value', 0.0, 1.0, 0.0, step=0.20, key="trainability_breed")
    st.slider('Demeanor Value', 0.0, 1.0, 0.0, step=0.20, key="demeanor_breed")
    
        # for breed prediction based on group
    groups_kwag = ['Foundation Stock Service', 'Terrier', 'Sporting', 'Working', 'Herding', 'Hound', 'Toy', 'Non-Sporting', 'Miscellaneous']
    st.selectbox('Select your type of dog', options=groups_kwag,key="group_type")

    if st.button('See the Result!', key='submit_2'):
        user_input_breed = get_breed_features()

        # for breed prediction
        # Remove the index column from the DataFrame
        breed_user_input_no_index = user_input_breed.reset_index(drop=True)

        st.divider()

        # Display the DataFrame in Streamlit
        st.subheader("Your input")
        st.dataframe(breed_user_input_no_index, hide_index=True)
        # st.write(breed_user_input_no_index.to_html(index=False, escape=False), unsafe_allow_html=True)

        # prediction function call
        pred_breed = breed_prediction(breed_user_input_no_index)

        st.markdown("                                       ")
        # display results
        st.write("##### Predicted Breed:", dog_map_breed_df['breed'].iloc[pred_breed])

        st.markdown("                                       ")
        string = 'Explore Dog Breeds'
        st.write(f"Check out the **{string}** page to know more about this breed!")


# ***************************************************************************************************************
# main prediction function call
# reading the csv for mapping, group mapping and calling the function to fetch user inputs from slider for both
# group prediction and breed prediction
# ***************************************************************************************************************

# Taking the user inputs via sliders

    
