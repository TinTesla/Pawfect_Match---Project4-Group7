import pandas as pd
import streamlit as st
import plotly_express as px
import plotly.figure_factory as ff
import os
import json
import numpy as np
import loadconfig

st.set_page_config(layout="wide")

appconfig = loadconfig.Appconfig()

#evaluation_file = "Data/output/outputModel_evaluation_1691869977.json"
evaluation_file = appconfig.evalulation_file
other_model_evaluation_file = appconfig.other_evaluation_file
#file_path = os.getcwd() + '/' + evaluation_file

#prediction_evaluation_file = "Data/output/Validation_result_all_v_1691869977.json"


st.header("Model Summary")

#reading training history json for plotting loss and accuracy
#reading the model evaluation details to get the model name to create the validation result file
with open(evaluation_file, 'r') as file_obj:
    data = json.load(file_obj)

    accuracy_history = data['training_history'].get('val_accuracy')
    loss_history = data['training_history'].get('val_loss')
    model_name = data['model_name']
    optimizer = data['optimizer']
    learning_rate = data['learning_rate']
    created_date = data['created_date']
    class_names = data['class_names']
    epochs = len(accuracy_history)
    if model_name:
        model_name = model_name.replace('.h5', '').replace('.model', '').strip()
        prediction_evaluation_file = 'Data/output/validation_result_{0}.json'.format(model_name)

        #open the second evaulation file for other hyperparameter model to compare with the running model
        with open(other_model_evaluation_file, 'r') as other_file_obj:
            other_data = json.load(other_file_obj)

            other_accuracy_history = other_data['training_history'].get('val_accuracy')
            other_loss_history = other_data['training_history'].get('val_loss')
            other_model_name = other_data['model_name']
            other_optimizer = other_data['optimizer']
            other_learning_rate = other_data['learning_rate']
            other_created_date = other_data['created_date']
            other_class_names = other_data['class_names']
            other_epochs = len(other_accuracy_history)
            if other_model_name:
                other_model_name = other_model_name.replace('.h5', '').replace('.model', '').strip()
                other_prediction_evaluation_file = 'Data/output/validation_result_{0}.json'.format(other_model_name)


#displaying model name
st.subheader("Current Model: " + model_name)
st.markdown("**Open Colab Notebook [DogImageClassification](https://colab.research.google.com/drive/1V7ZqBh9gIULgE2riAVytltjVVHyeQtRj?usp=sharing)**")
st.markdown("Optimizer: " + optimizer)
st.markdown(created_date)

#displaying the comparison of expected class and predicted class with validation data
df = pd.read_json(prediction_evaluation_file)
df_other_model = pd.read_json(other_prediction_evaluation_file)

def getOutcome(row):
    expected_class = row.get('Expected_Class')
    predicted_class = row.get('Predicted_Class')
    outcome = "Match" if expected_class == predicted_class else "No Match"

    return outcome

if len(df) > 0:
    df['Outcome'] = df.apply(lambda x:getOutcome(row=x),axis=1)
    df_no_matches = df[df['Outcome']=="No Match"]
    df_no_match = pd.DataFrame(df_no_matches.groupby(['Expected_Class','Predicted_Class'])['Image_url'].count()).reset_index()
    df_no_match.rename(columns={"Image_url":"Image_count"},inplace=True)
    df_result_counts = pd.DataFrame(df.groupby(['Expected_Class','Outcome'])['Image_url'].count()).reset_index()
    df_result_counts.rename(columns={"Image_url":"Image_count"},inplace=True)

    #for model with other optimizer
    df_other_model['Outcome'] = df_other_model.apply(lambda x:getOutcome(row=x),axis=1)
    df_other_no_matches = df_other_model[df_other_model['Outcome'] == "No Match"]
    df_other_no_matches = pd.DataFrame(df_other_no_matches.groupby(['Expected_Class', 'Predicted_Class'])['Image_url'].count()).reset_index()
    df_other_no_matches.rename(columns={"Image_url": "Image_count"}, inplace=True)
    df_other_result_counts = pd.DataFrame(df_other_model.groupby(['Expected_Class', 'Outcome'])['Image_url'].count()).reset_index()
    df_other_result_counts.rename(columns={"Image_url": "Image_count"}, inplace=True)


    t1,t2,t3,t4 = st.tabs(["**Prediction Outcome Chart**","**Prediction misses**","**Accuracy and Loss**","**Data**"])

    with t1:
        ct1,ct2 = st.columns([1,1])
        with ct1:

            st.markdown("**" + model_name +"**"+ " | " + optimizer + " optimizer | learning rate "+str(learning_rate) )
            fig = px.bar(df_result_counts,y="Expected_Class",x='Image_count',color='Outcome',
                         title="Prediction outcome per breed",orientation="h",height=800)
            st.plotly_chart(fig,use_container_width=True)

        with ct2:

            st.markdown("**" + other_model_name +"**"+ " | " + other_optimizer + " optimizer | learning rate "+str(other_learning_rate) )
            fig_o = px.bar(df_other_result_counts, y="Expected_Class", x='Image_count', color='Outcome',
                         title="Prediction outcome per breed", orientation="h", height=800)
            st.plotly_chart(fig_o, use_container_width=True)

    with t2:
        cd1,cd2 = st.columns([1,1])
        with cd1:
            st.markdown("**" + model_name + "**" + " | " + optimizer + " optimizer | learning rate " + str(learning_rate))
            fig2 = px.density_heatmap(df_no_match,y="Predicted_Class",x="Expected_Class",z='Image_count',height=800,
                                      title="Expected Dog Breed vs Predicted Dog Breed")
            fig2.update(layout_coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        with cd2:
            st.markdown("**" + other_model_name + "**" + " | " + other_optimizer + " optimizer | learning rate " + str(
                other_learning_rate))
            fig2_o = px.density_heatmap(df_other_no_matches,y="Predicted_Class",x="Expected_Class",z='Image_count',height=800,
                                      title="Expected Dog Breed vs Predicted Dog Breed")
            fig2_o.update(layout_coloraxis_showscale=False)
            st.plotly_chart(fig2_o, use_container_width=True)

    with t3:


    #for training history plots


        epochs = np.arange(0,epochs)
        other_epochs = np.arange(0,other_epochs)

        c1,c2,c3,c4 = st.columns([1,1,1,1])

        with c1:

            fig = px.line(x=epochs, y=accuracy_history, title='Accuracy vs Epochs (SGD optimizer)', labels={'x': 'epoch', 'y': 'accuracy'}
                          ,width=350)
            st.plotly_chart(fig)
        with c2:
            fig2 = px.line(x=epochs, y=loss_history, title='Loss vs Epochs (SGD optimizer)', labels={'x': 'epoch', 'y': 'loss'},
                           width=350)
            st.plotly_chart(fig2)

        with c3:
            fig3 = px.line(x=other_epochs, y=other_accuracy_history, title='Accuracy vs Epochs (Adam optimizer)', labels={'x': 'epoch', 'y': 'accuracy'}
                          ,width=350)
            st.plotly_chart(fig3)
        with c4:
            fig4 = px.line(x=other_epochs, y=other_loss_history, title='Loss vs Epochs (Adam optimizer)', labels={'x': 'epoch', 'y': 'loss'},
                           width=350)
            st.plotly_chart(fig4)



    with t4:

        cf1,cf2 = st.columns([1,1])
        with cf1:
            st.markdown("**" + model_name + "**" + " | " + optimizer + " optimizer | learning rate " + str(learning_rate))
            st.dataframe(df_result_counts)
        with cf2:
            st.markdown("**" + other_model_name + "**" + " | " + other_optimizer + " optimizer | learning rate " + str(
                other_learning_rate))
            st.dataframe(df_other_result_counts)