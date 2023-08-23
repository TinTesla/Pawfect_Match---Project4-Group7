import streamlit as st
import pandas as pd
#import numpy as np
import plotly_express as px
# import folium
# from streamlit_folium import st_folium

#st.set_page_config(layout="wide")

def parse_int(text):
    nbr = None
    try:
        nbr = int(text)
    except:
        nbr = None
    return nbr

def submitInput():
    zipInput = st.session_state.get('zipInput')
    if zipInput in zip_df['ZIP'].values:
        mask = zip_df['ZIP'] == zipInput
        userlat = zip_df.loc[mask,'LAT'].iloc[0]
        userlon = zip_df.loc[mask,'LNG'].iloc[0]
    else: # set to default location (berkeley,ca,usa)
        userlat = 37.871593
        userlon = -122.272743
        selected = 94704
    fig = map_func(zipInput, userlat, userlon)
    st.plotly_chart(fig)

def getZipforCity(location):
    zip_list = []
    shelters_df_location = shelters_df[shelters_df['location']==location]

    if len(shelters_df_location) > 0:
        zip_list = list(shelters_df_location['zip'].astype(int).drop_duplicates())

        zip_list.sort()

    return zip_list

def map_func(startzip, startlat, startlon):
    """ Generate a folium map, display with st_folium"""
    # create map
    #m = folium.Map(location=[startlat, startlon], zoom_start=10)

    shelters_df['zip'] = shelters_df['zip'].apply(parse_int)
    shelters_df_filtered = shelters_df[pd.notnull(shelters_df['zip'])]
    shelters_df_filtered = shelters_df_filtered[(shelters_df_filtered['zip'] > (startzip-1000)) & (shelters_df_filtered['zip'] < (startzip + 1000)) & (shelters_df_filtered['country'] == 'US')]
    shelters_df_filtered['size'] = 5

    site_lat = shelters_df_filtered['latitude']
    site_lon = shelters_df_filtered['longitude']
    locations_name = shelters_df_filtered['name']


    fig = px.scatter_mapbox(shelters_df_filtered, lat="latitude", lon="longitude", hover_name="name",hover_data=['city'],mapbox_style="open-street-map",width=725,height=800,
                            color_discrete_sequence=["blue"],size ='size',zoom=6,opacity=0.5,title='Adoption Shelters around you')

    fig.update_layout(margin={"r": 0, "l": 0, "b": 0},showlegend=False)
    return fig



# read in animal shelter csv into df
file_path_s = "Data/input/petfinder_shelters.csv"
shelters_df = pd.read_csv(file_path_s, sep=',')
# address1,address2,city,country,email,id,latitude,longitude,name,phone,state,zip
shelters_df['location'] = shelters_df['city'] + ' (' + shelters_df['state'] + ')'
cities = list(shelters_df[shelters_df['country']=='US']['location'].drop_duplicates().sort_values())

#read in zipcodes & lat/lon csv into df
file_path_z = "Data/input/zip_lat_long.csv"
zip_df = pd.read_csv(file_path_z, sep=',')
# ZIP,LAT,LNG

#start up page
st.set_page_config(
    page_title="Find a Pup Near You!",
    layout="centered",
    initial_sidebar_state="auto"
)

st.header(':green[Find a Pup Near You! :dog2:]')
st.sidebar.header(':green[Find a Pup Near You! :dog2:]')


# select city - dropdown
select_city = st.selectbox(label='Select City',options=cities)

if select_city:
    city_zips = getZipforCity(location=select_city)
    if len(city_zips) > 0:

        selected = st.number_input("Search by zipcode...",
                                       min_value=city_zips[0], max_value = city_zips[-1],
                                       value=city_zips[0],
                                       help=':sparkles: Accepts US zipcodes from {0} to {1} :sparkles:'.format(str(city_zips[0]),str(city_zips[-1])),
                                       key='zipInput'
                                       )
        if selected:
            if selected in zip_df['ZIP'].values:
                mask = zip_df['ZIP'] == selected
                userlat = zip_df.loc[mask, 'LAT'].iloc[0]
                userlon = zip_df.loc[mask, 'LNG'].iloc[0]
            else:  # set to default location (berkeley,ca,usa)
                userlat = 37.871593
                userlon = -122.272743
                selected = 94704

            fig = map_func(selected, userlat, userlon)
            st.plotly_chart(fig)


    else:
        st.warning("No shelters found")

