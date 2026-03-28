import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"

st.title('Machine learning app')

st.info('This app builds a machine learning model!')

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    X_raw

    st.write('**Y**')
    Y_raw = df['species']
    Y_raw
    

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')   

#Data preparations

with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', options=X_raw['island'].unique())
    gender = st.selectbox('Gender', options=X_raw['sex'].unique())
    bill_length_mm = st.slider('Bill length (mm)', min_value=int(X_raw['bill_length_mm'].min()), max_value=int(X_raw['bill_length_mm'].max()), value=int(X_raw['bill_length_mm'].mean()))
    bill_depth_mm = st.slider('Bill depth (mm)', min_value=int(X_raw['bill_depth_mm'].min()), max_value=int(X_raw['bill_depth_mm'].max()), value=int(X_raw['bill_depth_mm'].mean()))
    flipper_length_mm = st.slider('Flipper length (mm)', min_value=int(X_raw['flipper_length_mm'].min()), max_value=int(X_raw['flipper_length_mm'].max()), value=int(X_raw['flipper_length_mm'].mean()))
    body_mass_g = st.slider('Body mass (g)', min_value=int(X_raw['body_mass_g'].min()), max_value=int(X_raw['body_mass_g'].max()), value=int(X_raw['body_mass_g'].mean()))  

    
# Create Dataframe for the input features

data ={
    'island': island,
    'sex': gender,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    input_df
    st.write('**Input penguins with the whole dataset**')
    input_penguins


# Encode x

encode =['island', 'sex']
df_pens = pd.get_dummies(input_penguins, columns=encode)

X = df_pens[1:]

input_row = df_pens[:1]

# Encode y

target_mapper ={'Adelie': 0, 
                'Chinstrap': 1, 
                'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]

y = Y_raw.apply(target_encode)


with st.expander('Data preparation'):
    st.write('**Encoded X (input penguins)**')
    input_row
    st.write('**Encoded y**')
    y

# Model training and influence of the input features
# Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)
prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])


# Display predicted species
st.subheader('Predicted species')
st.dataframe(prediction_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn(
                     'Adelie',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                     ),
                 'Chinstrap': st.column_config.ProgressColumn(
                     'Chinstrap',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                     ),
                 'Gentoo': st.column_config.ProgressColumn(
                     'Gentoo',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                     )
             })

penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(penguin_species[prediction][0])