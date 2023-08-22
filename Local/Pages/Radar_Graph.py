# dependancies
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# imports
scaled_df = pd.read_csv("../Data/Output/scaled_df.csv")
grouped_scaled_df = pd.read_csv("../Data/Output/grouped_scaled_df.csv")

# page setup
def main():
    #title
    st.title("Selected Breed vs Breed Group")

    # drop-down
    selected_breed = st.selectbox("Select a Breed", scaled_df["Breed"])

    # selection
    breed_selection = scaled_df[scaled_df["Breed"] == selected_breed].iloc[0]

    # selection's group
    group_selection = breed_selection["Breed Group"]

    # selection's group's metrics
    group_row = grouped_scaled_df[grouped_scaled_df["Breed Group"] == group_selection].iloc[0]

    # selection's group plot (first to have behind)
    fig = px.line_polar(
        r=group_row[1:],
        theta=group_row.index[1:],
        title=f"Radar Chart - {group_selection} Metrics",
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
            title=f"Radar Chart - {selected_breed} Metrics",
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
    fig.update_layout(annotations=annotations)

    # print
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
