# dependancies
import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json


st.set_page_config(layout="wide")
st.markdown('# Explore Dog Breeds 🔍')
st.sidebar.markdown('# Explore Dog Breeds 🔍')

root_directory = os.getcwd()
db_folder = 'db'
db_name = 'dogdata.db'
db_path = root_directory + '/' + db_folder + '/' + db_name

sqliteConn = sqlite3.connect(db_path)


# imports
scaled_df = pd.read_csv("Data/output/scaled_df.csv")
grouped_scaled_df = pd.read_csv("Data/output/grouped_scaled_df.csv")

#for database connection : fetching AKC data for the selected breed
def get_species_info(species_name):
    print(species_name)
    query = "Select * from Dog_Info_AKC where Dog_Species = '{0}';".format(species_name)
    df_species_info = pd.read_sql(query,con=sqliteConn)
    return df_species_info


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


# page setup
def main():
    #title
    #st.title("Explore by Breed")

    # drop-down
    selected_breed = st.selectbox("Select a Breed", scaled_df["Breed"])
    print("selected breed", selected_breed)


    #fetch AKC data for the selected species
    df_akc_output = get_species_info(species_name=selected_breed)

    # selection
    breed_selection = scaled_df[scaled_df["Breed"] == selected_breed].iloc[0]
    average_height = breed_selection['Height (avg)']
    average_weight = breed_selection['Weight (avg)']
    average_lifespan = breed_selection['Lifespan (avg)']

    # selection's group
    group_selection = breed_selection["Breed Group"]

    # selection's group's metrics
    group_row = grouped_scaled_df[grouped_scaled_df["Breed Group"] == group_selection].iloc[0]

    # selection's group plot (first to have behind)
    fig = px.line_polar(
        r=group_row[1:],
        theta=group_row.index[1:],
        #title=f" {group_selection} Metrics",
        line_close=True,
        color_discrete_sequence=['blue'],
        line_dash_sequence=['solid'],
        labels={'r': 'Metric Value', 'theta': 'Metric'}
    )

    # selection's plot (second to be on top)
    fig.add_trace(
        px.line_polar(
            r=breed_selection[2:],
            theta=breed_selection.index[2:],
            #title=f" {selected_breed} Metrics",
            line_close=True,
            color_discrete_sequence=['red'],
            line_dash_sequence=['solid'],
            labels={'r': 'Metric Value', 'theta': 'Metric'}
        ).data[0]
    )

    # plot fill
    fig.update_traces(fill='toself')

    # adding labels manually
    annotations = [
        go.layout.Annotation(
            x=1.00,
            y=-0.2,
            text=f'Red = {selected_breed}',
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=15, color='red')
        ),
        go.layout.Annotation(
            x=1.00,
            y=-0.3,
            text=f'Blue = {group_selection}',
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=15, color='blue')
        )
    ]

    # pushing labels
    fig.update_layout(annotations=annotations,width=500,height=400)

    co1,co2 = st.columns([1,2])

    with co1:


        if len(df_akc_output) > 0:

            akc_data = df_akc_output.to_json(orient='records')
            akc_data = json.loads(akc_data)
            akc = akc_data[0]

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
            species_image = akc.get('Img_Link')

            st.image(species_image,use_column_width=True)

            if temperament:
                st.markdown( "#### " + temperament)

            if energy_level_value:
                energyIcon = getValueIcon(icon_type='Energy Level', icon_level_val=energy_level_value,
                                          icon_category_text=energy_level_category)
                st.markdown(energyIcon)

            if demeanor_value:
                demeanorIcon = getValueIcon(icon_type='Demeanor', icon_level_val=demeanor_value,
                                            icon_category_text=demeanor_category)
                st.markdown(demeanorIcon)

            if trainability_value:
                trainIcon = getValueIcon(icon_type='Trainability', icon_level_val=trainability_value,
                                         icon_category_text=trainability_category)
                st.markdown(trainIcon)

            if grooming_frequency_value:
                groomIcon = getValueIcon(icon_type='Grooming', icon_level_val=grooming_frequency_value,
                                         icon_category_text=grooming_frequency_category)
                st.markdown(groomIcon)

            if shedding_value:
                shedIcon = getValueIcon(icon_type='Shedding', icon_level_val=shedding_value,
                                         icon_category_text=shedding_category)
                st.markdown(shedIcon)



    with co2:
        # print
        st.subheader("Breed vs Group Average")
        st.markdown("##### " + selected_breed + " | " + group_selection)

        st.plotly_chart(fig)

        st.slider(label="##### Average Height :dog: ", min_value=0.0, max_value=1.0, value=average_height,
                  disabled=True)
        st.slider(label="##### Average Weight :rock: ", min_value=0.0, max_value=1.0, value=average_weight,
                  disabled=True)

        st.slider(label="##### Average Lifespan :heavy_heart_exclamation_mark_ornament: ", min_value=0.0,
                  max_value=1.0,
                  value=average_lifespan, disabled=True)




#if __name__ == "__main__":
main()
