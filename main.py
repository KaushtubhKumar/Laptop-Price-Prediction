# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# pipe=pickle.load(open("pipe.pkl","rb"))
# df=pickle.load(open('df.pkl','rb'))
#
# st.title('Laptop predictor')
#
# #brand
# company=st.selectbox('Brand',df['Company'].unique())
#
# #type of laptop
# types=st.selectbox('Type',df['TypeName'].unique())
#
# #ram
# ram=st.selectbox('RAM(IN Gb)',[2,4,6,8,12,16,32,64])
#
# weight=st.number_input('Weight of the Laptop')
#
# touchscreen=st.selectbox('Touchscreen',['Yes','No'])
#
# ips=st.selectbox('IPS-display',['Yes','No'])
#
# screen_size=st.number_input('Screen Size')
#
# resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
#
# cpu=st.selectbox('CPU',df['cpu_brand'].unique())
# hdd=st.selectbox('HDD(IN GB)',[0,128,256,512,1024,2048])
# ssd=st.selectbox('SDD(IN GB)',[0,8,16,64,128,256,512,1024])
# gpu=st.selectbox('GPU',df['gpus'].unique())
# os=st.selectbox('OS',df['OS'].unique())
#
# if st.button('Predict Price'):
#     if touchscreen=='Yes':
#         touchscreen=1
#     else:
#         touchscreen=0
#
#     if ips=='Yes':
#         ips=1
#     else:
#         ips=0
#
# x_res=int(resolution.split('x')[0])
# y_res=int(resolution.split('x')[1])
# ppi=np.sqrt(x_res**2+y_res**2)/screen_size
#
# # query=np.array([company,types,ram,weight,touchscreen,ips,screen_size,cpu,hdd,ssd,gpu,os])
# #
# # query=query.reshape(1,-1)
# # prediction=pipe.predict(query)
# # st.title(np.exp(prediction))
#
# # Create a DataFrame with the correct column names and types
# query = pd.DataFrame([[company, types, ram, weight, touchscreen, ips, screen_size, cpu, hdd, ssd, gpu, os]],
#                      columns=['Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen', 'ips', 'ppi', 'cpu_brand', 'HDD', 'SSD', 'gpus', 'OS'])
#
# # prediction = pipe.predict(query) # Use the DataFrame directly
# st.title(f"The predicted price is: ₹{int(np.exp(pipe.predict(query)[0]))}")

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and dataframe
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Price Predictor')

# 1. Brand
company = st.selectbox('Brand', df['Company'].unique())

# 2. Type
types = st.selectbox('Type', df['TypeName'].unique())

# 3. RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# 4. Weight
# FIX: Removed '<=10' so it returns the actual number (e.g. 1.5), not True/False
weight = st.number_input('Weight of the Laptop')

# 5. Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# 6. IPS
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# 7. Screen Size (Needed for PPI calculation)
screen_size = st.number_input('Screen Size (in Inches)')

# 8. Resolution (Needed for PPI calculation)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# 9. CPU
cpu = st.selectbox('CPU', df['cpu_brand'].unique())

# 10. HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# 11. SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# 12. GPU
gpu = st.selectbox('GPU', df['gpus'].unique())

# 13. OS
os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):
    # 1. Convert Yes/No to 1/0
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # 2. Calculate PPI (Pixels Per Inch)
    # The model was trained on 'ppi', not 'screen_size'
    if screen_size > 0:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    else:
        ppi = 0  # Avoid division by zero

    # 3. Create DataFrame with the EXACT column order used in training
    # The order deduced from your notebook is:
    # [Company, TypeName, Ram, Weight, TouchScreen, Ips, ppi, cpu_brand, gpus, HDD, SSD, OS]

    query = pd.DataFrame([[company, types, ram, weight, touchscreen, ips, ppi, cpu, gpu, hdd, ssd, os]],
                         columns=['Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen', 'ips', 'ppi',
                                  'cpu_brand', 'gpus', 'HDD', 'SSD', 'OS'])

    # 4. Predict
    # We use np.exp() because the model was likely trained on log-transformed prices
    try:
        prediction = pipe.predict(query)
        st.title(f"The predicted price is: ₹{int(np.exp(prediction[0]))}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")