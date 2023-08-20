import streamlit as st
import pandas as pd
#import numpy as np
import folium
from streamlit_folium import st_folium

def map_func(startzip, startlat, startlon):
    """ Generate a folium map, display with st_folium"""
    # create map
    m = folium.Map(location=[startlat, startlon], zoom_start=10)

    # create loop to write markers for every shelter in ~100 mi box
    for lat,lon,name,phone,email,city,zipcode,country in zip(shelters_df['latitude'], shelters_df['longitude'], shelters_df['name'], 
                                                             shelters_df['phone'], shelters_df['email'], shelters_df['city'], shelters_df['zip'],shelters_df['country']):
        # check to see if zipcode is +- 1000
        if (country == 'US'):
            zipc = int(zipcode)
            if (zipc > (startzip-1000)) and (zipc < (startzip + 1000)):
                # get info for popup name, address, phone, email
                pop_box = name
                zipc = str(zipcode)
                # input city, zipcode
                if(type(city)==str):
                    address ='\n' + city + ', ' + zipc
                else:
                    address ='\n' + zipc
                
                pop_box = pop_box + address
                # input email
                if (type(email)==str):
                    pop_box = pop_box + '\n' + email
                # input phone number
                if(type(phone)==str):
                    pop_box = pop_box + '\n' + phone
                # pop_box updated
                # add marker
                folium.Marker(location=[lat,lon], tooltip = name, popup = pop_box).add_to(m)
                # end for loop`
    # call to render Folium map in Streamlit
    return st_folium(m, width=725)

# read in animal shelter csv into df
file_path_s = "Data/Input/petfinder_shelters.csv"
shelters_df = pd.read_csv(file_path_s, sep=',')
# address1,address2,city,country,email,id,latitude,longitude,name,phone,state,zip

#read in zipcodes & lat/lon csv into df
file_path_z = "Data/Input/zip_lat_long.csv"
zip_df = pd.read_csv(file_path_z, sep=',')
# ZIP,LAT,LNG

#start up page
st.set_page_config(
    page_title="Find a Pup Near You!",
    layout="centered",
    initial_sidebar_state="auto"
)

st.header(':green[Find a Pup Near You! :dog2:]')

with st.form(key='map_key'):
    selected = st.number_input("Search by zipcode...", min_value=601, max_value = 99929, value=94704, help=':sparkles: Accepts US zipcodes from 00601 to 99929 :sparkles:')
    st.write('The current zipcode is: ', selected)
    sub = st.form_submit_button('Generate Map')

# if zipcode was input to search
if selected:
    if selected in zip_df['ZIP'].values:
        mask = zip_df['ZIP'] == selected
        userlat = zip_df.loc[mask,'LAT'].iloc[0]
        userlon = zip_df.loc[mask,'LNG'].iloc[0]
    else: # set to default location (berkeley,ca,usa)
        userlat = 37.871593
        userlon = -122.272743
        selected = 94704
    
    st_data = map_func(selected, userlat, userlon)
else:
    st.write('Page Error: 404 Perra')