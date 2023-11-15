#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import numpy as np
import pickle
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


# In[13]:


filename = r"C:\Users\nurul asyrifah\Desktop\Jupyter\churn.sav"


# In[14]:


loaded_model = pickle.load(open(filename, "rb"))


# In[15]:


scaler_filename=r"C:\Users\nurul asyrifah\Desktop\Jupyter\scaler.sav"


# In[16]:


loaded_scaler=pickle.load(open(scaler_filename,"rb"))


# In[17]:


mapping1 = {'Jakarta': 1, 'Bandung': 2}
mapping2 = {'Mid End': 1, 'High End': 2, 'Low End': 3}
mapping3 = {'Yes': 1, 'No': 2, 'No internet service': 3}
mapping4 = {'No': 1, 'Yes': 2}
mapping5 = {'Digital Wallet': 1, 'Pulsa': 2, 'Debit': 3, 'Credit': 4}
mapping6 = {'Yes': 1, 'No': 0}


# In[18]:


st.set_page_config(
    page_title="Churn Prediction",
    page_icon=":chart_with_upwards_trend:"
)


# In[19]:


# Streamlit app
st.title('Churn Prediction App')


# In[24]:


# Create input fields for user input
st.sidebar.header('Input Customer Data')
tenure_months = st.sidebar.number_input('Tenure Months')
location = st.sidebar.selectbox('Location', list(mapping1.keys()))
device_class = st.sidebar.selectbox('Device Class', list(mapping2.keys()))
games_product = st.sidebar.selectbox('Games Product', list(mapping3.keys()))
music_product = st.sidebar.selectbox('Music Product', list(mapping3.keys()))
education_product = st.sidebar.selectbox('Education Product', list(mapping3.keys()))
call_center = st.sidebar.selectbox('Call Center', list(mapping4.keys()))
video_product = st.sidebar.selectbox('Video Product', list(mapping3.keys()))
use_myapp = st.sidebar.selectbox('Use MyApp', list(mapping3.keys()))
payment_method = st.sidebar.selectbox('Payment Method', list(mapping5.keys()))
monthly_purchase_thou_idr = st.sidebar.number_input('Monthly Purchase (Thou. IDR)')


# In[25]:


# Map user input to label encoded values
location_encoded = mapping1.get(location, 0)
device_class_encoded = mapping2.get(device_class, 0)
games_product_encoded = mapping3.get(games_product, 0)
music_product_encoded = mapping3.get(music_product, 0)
education_product_encoded = mapping3.get(education_product, 0)
call_center_encoded = mapping4.get(call_center, 0)
video_product_encoded = mapping3.get(video_product, 0)
use_myapp_encoded = mapping3.get(use_myapp, 0)
payment_method_encoded = mapping5.get(payment_method, 0)


# In[26]:


# Prepare input data for prediction
input_data = [tenure_months, location_encoded, device_class_encoded, games_product_encoded, music_product_encoded,
              education_product_encoded, call_center_encoded, video_product_encoded, use_myapp_encoded,
              payment_method_encoded, monthly_purchase_thou_idr]


# In[27]:


# Add a "Predict" button
if st.sidebar.button('Predict'):
    input_data_array = np.array(input_data).reshape(1, -1)

    # Scale the input data using the loaded StandardScaler
    std_data = loaded_scaler.transform(input_data_array)

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(std_data)

    # Display the prediction result
    st.header('Churn Prediction Result:')
    if prediction[0] == 0:
        st.write("Customer is predicted as 'Not Churn'")
    else:
        st.write("Customer is predicted as 'Churn'")
else: 
    st.header('Churn:')
    st.markdown("<p style='text-align: justify;'>The term 'churn' refers to a concept in business that pertains to the number of customers or clients who stop using a company's products or services within a specific period. Churn is commonly used in the context of subscription services, such as internet subscriptions, cable television, mobile phones, streaming platforms, and other businesses that involve customers paying periodically. This term is generally employed in subscription-based industries where customers pay for services on a recurring basis. A high churn rate can be a concern for companies, as it may indicate issues in retaining customers or customer dissatisfaction.</p>", unsafe_allow_html=True)
    st.write("")
    st.write("<p style='text-align: justify;'>Companies typically strive to reduce churn rates through customer retention strategies, improving service quality, and responding to customer feedback. Churn analysis can also provide insights into the factors influencing customers' decisions to stop using a company's products or services.</p>", unsafe_allow_html=True)


# In[31]:





# In[ ]:




