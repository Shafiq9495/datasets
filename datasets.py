import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
st.markdown('<style>body{background-color: #E8E8E8;}</style>',unsafe_allow_html=True)

url = 'http://localhost/ovi/admin-dashboard.php'

st.title("Online Vehicle Identification")
st.subheader("Datasets")
if st.button('Back'):
    webbrowser.open_new_tab(url)


@st.cache
def load_data(choise_ds):
   data = pd.read_csv('Traffic_Violations_ds.csv', low_memory=False)
   del data['Date Of Stop']
   del data['Time Of Stop']
   data.dropna(axis=0, subset=['Latitude'], inplace=True)
   data.dropna(axis=0, subset=['Longitude'], inplace=True)
   data.dropna(axis=0, subset=['Year'], inplace=True)
   data.dropna(axis=0, subset=['Article'], inplace=True)
   data.dropna(axis=0, subset=['Geolocation'], inplace=True)
   data['Description'].fillna('Other', inplace = True)
   data['Location'].fillna('Other', inplace = True)
   data['State'].fillna('Other', inplace = True)
   data['Make'].fillna('Other', inplace = True)
   data['Model'].fillna('Other', inplace = True)
   data['Color'].fillna('Other', inplace = True)
   data['Driver City'].fillna('Other', inplace = True)
   data['Driver State'].fillna('Other', inplace = True)
   data['DL State'].fillna('Other', inplace = True)
   return data

@st.cache
def load_data2(parking_data):
   data = pd.read_csv('Parking_Violations_Issued.csv', low_memory=False)
   del data['Summons Number']
   data['Violation In Front Of Or Opposite'].fillna('F', inplace = True)
   data.fillna('other', inplace = True)
   return data

choice_ds = st.selectbox("Select Datasets:",("Traffic Violations Issued","Parking Violations Issued"))

df = load_data(choice_ds)
df_fs = df.copy()
df_fs = df_fs.head(10000)
dataset = df_fs.copy()

# ->2. Parking Violations Datasets 
data_parking = load_data2(choice_ds)
dset =data_parking.head(10000)
df_new = data_parking.copy()
df_ds = data_parking.copy()

if (choice_ds == 'Traffic Violations Issued'):
    st.subheader("Traffic Violations Datasets")
    st.write(df_fs.head(20))
    st.write("Shape of dataset:", df_fs.shape)
else:
    st.subheader("Parking Violations Datasets")
    st.write(dset.head(20))
    st.write("Shape of dataset:", dset.shape)