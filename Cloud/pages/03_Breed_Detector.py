import pandas as pd
import streamlit as st
import Predict_Breed as pbr
import time
import os
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np

st.set_page_config(layout="wide")

root_directory = os.getcwd()
user_data_directory = 'User_Data'
display_image_directory = 'Data/Display_Dog_Breeds'
user_data_path = root_directory + '/' + user_data_directory



#for database
db_folder = 'db'
db_name = 'dogdata.db'
db_path = root_directory + '/' + db_folder + '/' + db_name

sqliteConn = sqlite3.connect(db_path)



@st.cache_data

def getSpeciesPrediction(image_file):

    return img_predictor.getPredictions(image_file=image_file)


def getValueIcon(icon_type,icon_level_val,icon_category_text):
    icon = ''
    icon_emoji = ''
    if icon_type == 'Demeanor':
        icon_emoji = ':heart_decoration: '
    elif icon_type == 'Energy Level':
        icon_emoji = ':boom::collision: '
    elif icon_type == 'Trainability':
        icon_emoji = ':eight_pointed_black_star: '
    elif icon_type == 'Grooming':
        icon_emoji = ':scissors: '
    elif icon_type == 'Shedding':
        icon_emoji = ':sheep: '


    if icon_level_val > 0.0 and icon_level_val <= 0.25:
        icon = icon_emoji + icon_category_text
    elif icon_level_val > 0.25 and icon_level_val <= 0.5:
        icon = icon_emoji + icon_emoji + icon_category_text
    elif icon_level_val > 0.5 and icon_level_val <= 0.75:
        icon = icon_emoji + icon_emoji + icon_emoji + icon_category_text
    else:
        icon = icon_emoji + icon_emoji + icon_emoji + icon_emoji + icon_emoji + icon_category_text

    return icon


#get breed look up
def get_breed_lookup():

    query = "Select * from Dog_Info_AKC"
    df_breed_info = pd.read_sql(query, con=sqliteConn)
    df_breed_info['Key'] = df_breed_info['Dog_Species']
    df_breed_info.set_index('Key',inplace=True)
    df_breed_lookup = df_breed_info.to_json(orient='index')
    df_breed_lookup = json.loads(df_breed_lookup)

    return df_breed_lookup




def get_species_info(species_name):

    query = "Select * from Dog_Info_AKC where classlabel = '{0}';".format(species_name)
    df_species_info = pd.read_sql(query,con=sqliteConn)
    return df_species_info

#fetching species group other than the predicted breed
def get_same_group_breed(species_name):

    query_file = 'sql/breed_groups.sql'
    with open(query_file,'r') as file_obj:
        query_text = file_obj.read()
        query_text = query_text.replace('$breed_name',species_name)
        df_groups = pd.read_sql(query_text,con=sqliteConn)

        return df_groups

#get breed name from label
def get_breed_from_index(sim_idx,breed_labels):
    breed_name = None
    if sim_idx <= len(breed_labels):
        breed_name = breed_labels[sim_idx]
    return breed_name



#calculate similarity one record and a list of records with numeric values only (dog breed characteristics)
#related dog breeds of same group(Top N)

def calculate_breed_similarity(predicted_breed_features,all_breed_features,breed_labels):

    similarity = cosine_similarity(predicted_breed_features,all_breed_features,dense_output=True)
    #similarity.argsort()[-5:][::-1]
    similarity = similarity.tolist()[0]

    similarity_data = []
    similarity_output = []
    for sim in similarity:
        sim_score = sim
        sim_idx = similarity.index(sim)
        sim_obj = dict(sim_idx = sim_idx,score = sim_score)
        similarity_data.append(sim_obj)

    if len(similarity_data) > 0:
        df_similar_breeds = pd.DataFrame(similarity_data)
        df_similar_breeds['similarity_rank'] = df_similar_breeds['score'].rank(ascending=False)
        df_similar_breeds = df_similar_breeds[df_similar_breeds['similarity_rank'] <= 5]
        df_similar_breeds['breed_name'] = df_similar_breeds['sim_idx'].apply(lambda x:get_breed_from_index(sim_idx=x,breed_labels=breed_labels))
        df_similar_breeds['breed_info'] = df_similar_breeds['breed_name'].apply(lambda x:breed_lookup.get(x))
        df_similar_breeds.sort_values('similarity_rank',inplace=True)

        similarity_output = df_similar_breeds.to_json(orient='records')
        similarity_output = json.loads(similarity_output)

    return similarity_output

#getting similar breed picture for display on the page(not used)
def get_breed_images(breed_name):
    breed_image_folder = display_image_directory + '/' + breed_name
    breed_images = os.listdir(breed_image_folder)
    breed_image = breed_image_folder + '/' + breed_images[0] if len(breed_images) > 0 else None


    return breed_image


img_predictor = pbr.imagePredictor()

breed_lookup = get_breed_lookup()
comparison_cols = ["max_height","energy_level_value","trainability_value","demeanor_value"]

st.header('Welcome to Dog Breed Detector')
st.subheader('Please upload a dog picture')



upl = st.file_uploader(label='Upload image',type=['.jpg','.png','.jpeg'],accept_multiple_files=False)

breed_found = False

if upl is not None:
    bytes_data = upl.getvalue()

    image_file_name = upl.name
    image_file = user_data_path + '/' + image_file_name

    c1,c2 = st.columns([1,2])
    with c1:
        st.image(bytes_data, use_column_width="always")
        with open(image_file, 'wb') as img_obj:
            img_obj.write(bytes_data)
    predictions = getSpeciesPrediction(image_file=image_file)
    with c2:

        if len(predictions) > 0:

            top_prediction = predictions[0]
            species_name = top_prediction.get('Species')
            predicted_species_name = top_prediction.get('Species')
            pct_predicted = top_prediction.get('Predicted_percent')
            print("Species Name: ", species_name)
            print("Predicted % :", pct_predicted)
            #st.subheader("Predicted breed: " + species_name)
            # species_url = 'http://localhost:8501/Compatibility/?breed=' + species_name
            #
            # species_url_markdown = '**[See breed compatibility]({0})**'.format(species_url)

            df_akc_output = get_species_info(species_name=species_name)
            #print("akc output" + df_akc_output.to_json(orient="records", indent=4))
            if len(df_akc_output) > 0:
                breed_found = True
                # getting other breeds under same group
                df_same_group_breeds = get_same_group_breed(species_name=species_name)

                df_same_group_breeds.dropna(inplace=True)

                akc_data = df_akc_output.to_json(orient='records')
                akc_data = json.loads(akc_data)
                akc = akc_data[0]
                #for akc in akc_data:
                species_name = akc.get('Dog_Species')
                species_desc = akc.get('description')
                temperament = akc.get('temperament')
                species_group = akc.get('group')
                grooming_frequency_category = akc.get('grooming_frequency_category')
                grooming_frequency_value = akc.get('grooming_frequency_value')
                shedding_value = akc.get('shedding_value')
                shedding_category = akc.get('shedding_category')
                energy_level_value = akc.get('energy_level_value')
                energy_level_category = akc.get('energy_level_category')
                trainability_value = akc.get('trainability_value')
                trainability_category = akc.get('trainability_category')
                demeanor_value = akc.get('demeanor_value')
                demeanor_category = akc.get('demeanor_category')
                max_height = akc.get("max_height")
                max_weight = akc.get("max_weight")
                min_height = akc.get("min_height")
                min_weight = akc.get("min_weight")
                height_display = "No height Info"
                weight_display = "No weight Info"
                #for display of height and weight
                if min_height and max_height:
                    height_display = "###### Height: " + str(min_height) + " - " + str(max_height)
                if min_weight and max_weight:
                    weight_display = "###### Weight: " + str(min_weight) + " - " + str(max_weight) + " lbs"


                #detected breed details
                df_predicted_breed = df_akc_output[comparison_cols]
                df_predicted_breed = df_predicted_breed.values.tolist()

                st.subheader(species_name)
                st.markdown("Prediction % " + str(pct_predicted))

                st.markdown(species_group)

                with st.expander(label='Description'):
                    st.markdown(species_desc)

                cv1,cv2 = st.columns([1,1])
                with cv1:

                    if energy_level_value:
                        energyIcon = getValueIcon( icon_type='Energy Level',icon_level_val=energy_level_value,icon_category_text=energy_level_category)
                        st.markdown(energyIcon)

                    if demeanor_value:
                        demeanorIcon = getValueIcon(icon_type='Demeanor', icon_level_val=demeanor_value,
                                                    icon_category_text=demeanor_category)
                        st.markdown(demeanorIcon)

                    if trainability_value:
                        trainIcon = getValueIcon(icon_type='Trainability', icon_level_val=trainability_value,
                                                  icon_category_text=trainability_category)
                        st.markdown(trainIcon)


                with cv2:

                    if grooming_frequency_value:
                        groomIcon = getValueIcon(icon_type='Grooming', icon_level_val=grooming_frequency_value,
                                                 icon_category_text=grooming_frequency_category)
                        st.markdown(groomIcon)

                    if shedding_value:
                        shedIcon = getValueIcon(icon_type='Shedding', icon_level_val=shedding_value,
                                                icon_category_text=shedding_category)
                        st.markdown(shedIcon)

                    st.markdown(height_display)
                    st.markdown(weight_display)


            else:
                st.subheader("Predicted Breed: " + predicted_species_name)
                st.markdown("Prediction % " + str(pct_predicted))
                st.warning("No info found for species in AKC website")
        else:
            st.warning("Cannot identify breed")

    if breed_found:
        if len(df_same_group_breeds) > 0:

            breed_labels = list(df_same_group_breeds['Dog_Species'])
            df_same_groups = df_same_group_breeds[comparison_cols]
            df_same_groups = df_same_groups.values.tolist()

            # get similar breeds of same group as predicted breed
            similar_breed_val = calculate_breed_similarity(predicted_breed_features=df_predicted_breed,
                                                           all_breed_features=df_same_groups,
                                                           breed_labels=breed_labels)

            #print("Similar breed values", json.dumps(similar_breed_val, indent=4))

            if len(similar_breed_val) > 0:
                st.divider()
                st.subheader("Similar Breeds by traits within " + species_group)

                similar_breed_cols = st.columns(5)
                i = 0
                for similar_breed in similar_breed_val:
                    similar_breed_name = similar_breed.get("breed_name")

                    # smb_group = similar_breed.get("breed_info").get("group")
                    smb_temperament = similar_breed.get("breed_info").get("temperament")
                    smb_class = similar_breed.get("breed_info").get("classlabel")
                    smb_image = similar_breed.get("breed_info").get("Img_Link")
                    smb_grooming_frequency_value = similar_breed.get("breed_info").get("grooming_frequency_value")
                    smb_grooming_frequency_category = similar_breed.get("breed_info").get(
                        "grooming_frequency_category")
                    smb_shedding_value = similar_breed.get("breed_info").get(
                        "shedding_value")
                    smb_shedding_category = similar_breed.get("breed_info").get(
                        "shedding_category")
                    smb_energy_level_value = similar_breed.get("breed_info").get(
                        "energy_level_value")
                    smb_energy_level_category = similar_breed.get("breed_info").get(
                        "energy_level_category")
                    smb_trainability_value = similar_breed.get("breed_info").get(
                        "trainability_value")
                    smb_trainability_category = similar_breed.get("breed_info").get(
                        "trainability_category")
                    smb_demeanor_value = similar_breed.get("breed_info").get("demeanor_value")
                    smb_demeanor_category = similar_breed.get("breed_info").get("demeanor_category")

                    with similar_breed_cols[i]:
                        st.markdown("**" + similar_breed_name + "**")
                        # st.markdown(smb_group)
                        if smb_class:
                            if smb_image:
                               st.image(smb_image,width=320)
                        # breed characteristics
                        if smb_energy_level_value:
                            smb_energyIcon = getValueIcon(icon_type='Energy Level', icon_level_val=smb_energy_level_value,
                                                          icon_category_text=smb_energy_level_category)
                            st.markdown(smb_energyIcon)

                        if smb_trainability_value:
                            smb_trainIcon = getValueIcon(icon_type='Trainability', icon_level_val=smb_trainability_value,
                                                         icon_category_text=smb_trainability_category)
                            st.markdown(smb_trainIcon)

                        if smb_demeanor_value:
                            smb_demIcon = getValueIcon(icon_type='Demeanor',
                                                       icon_level_val=smb_demeanor_value,
                                                       icon_category_text=smb_demeanor_category)
                            st.markdown(smb_demIcon)
                    i = i + 1

        # st.dataframe(df_akc_output)
