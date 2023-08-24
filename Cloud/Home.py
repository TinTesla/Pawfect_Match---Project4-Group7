import streamlit as st
from PIL import Image
import sqlite3 as sqlite

# Homepage
st.set_page_config(page_title='Pawfect Match 🐶')
st.title('Pawfect Match 🐶')
st.sidebar.success('Select Any Page from here')

# ----------------------------------------------------------------
# Create the SQL connection to dogs_db
conn = sqlite.connect("dogs.db")
cur = conn.cursor()
# cur.execute("DROP TABLE IF EXISTS user_info")
# ----------------------------------------------------------------

def addData(a, b, c, d, e):
    cur.execute("""CREATE TABLE IF NOT EXISTS user_info (first_name TEXT(50), last_name TEXT(50), email TEXT(50), gender TEXT(20), here_for TEXT);""")
    cur.execute("""INSERT INTO user_info VALUES (?, ?, ?, ?, ?)""", (a, b, c, d, e))
    conn.commit()
    st.success('Successfully Submitted! Thank you for staying connected!')

st.markdown('# Home Page 🎈')
st.sidebar.markdown('# Home Page 🎈')

image = Image.open('image/dog.jpg')
st.image(image)

st.subheader('Looking for your Pawfect Match?')
st.write("Not sure which type of dog would be the best for you? Don't let this stop you from having the closest partner in your life! We got this :muscle:")
st.write('We are here to help: ')
st.write("""
    - Explore breed details
    - Upload a dog image to see which is the closest breed and other breeds in the same group
    - Finish the questionnaire to see which breed is best suitable for you
    - View the map to search dogs for adoption near you
""")
st.write('Before heading to our amazing feature pages, feel free to fill out the form to stay connected!')

st.subheader('More about you...')

with st.form('User Info Form', clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input('First Name')

    with col2:
        last_name = st.text_input('Last Name')

    email = st.text_input('Email Address📧')
    gender = st.radio('Gender🧑', ('Male', 'Female', 'Prefer Not To Say'))
    here_for = st.selectbox('How can we help?', ("I'm just looking around", "I am planning to get a dog", "I want to know more about dog breeds", "I have a dog picture and want to know the breed", "I want to know which kind of dog is the best for me", "I want to find a shelter"))

    submit_form = st.form_submit_button(label="Submit", help='Click to submit!')

    if submit_form:
        st.write(submit_form)

        if first_name and last_name and email and gender and here_for:
            st.success(
                f"User Info:  \n First Name: {first_name}  \n Last Name: {last_name}  \n Email: {email}  \n Gender: {gender}  \n Here_For: {here_for}"
            )

            addData(first_name, last_name, email, gender, here_for)

        else:
            st.warning('Please fill in all the fields.')

# db_table = cur.execute("SELECT * FROM user_info")

# st.dataframe(db_table.fetchall())
