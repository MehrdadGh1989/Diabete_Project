import streamlit as st
#import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
#import function
# C:\Users\ghadi\source\repos\Presentation_Diabete_Project_Streamlit\Presentation_Diabete_Project_Streamlit
# streamlit run Presentation_Diabete_Project_Streamlit.py
image=Image.open('C:/Users/ghadi/Desktop/Univr2.jpg')
st.image(image,use_column_width=True)
#Splitting the part in streamlit
st.write("""
***
""")
#Choosing Header in streamlit
st.header('Programming Project')
st.write('A**short** explanation about the Project')
#Explanation about the project


# How to Visualize (Justify)-TextFiles According to WordShape
 
Diabete_Explanation=st.markdown('<div style="text-align: justify;">Diabetes is among the most prevalent chronic diseases in the United States, impacting millions of Americans each year and exerting a significant financial burden on the economy. Diabetes is a serious chronic disease in which individuals lose the ability to effectively regulate levels of glucose in the blood, and can lead to reduced quality of life and life expectancy. After different foods are broken down into sugars during digestion, the sugars are then released into the bloodstream. This signals the pancreas to release insulin. Insulin helps enable cells within the body to use those sugars in the bloodstream for energy.Diabetes is generally characterized by either the body not making enough insulin or being unable to use the insulin that is made as effectively as needed.\n Complications like heart disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high levels of sugar remaining in the bloodstream for those with diabetes. While there is no cure for diabetes, strategies like losing weight, eating healthily, being active, and receiving medical treatments can mitigate the harms of this disease in many patients. Early diagnosis can lead to lifestyle changes and more effective treatment, making predictive models for diabetes risk important tools for public and public health officials.\nThe scale of this problem is also important to recognize. The Centers for Disease Control and Prevention has indicated that as of 2018, 34.2 million Americans have diabetes and 88 million have prediabetes. Furthermore, the CDC estimates that 1 in 5 diabetics, and roughly 8 in 10 prediabetics are unaware of their risk. While there are different types of diabetes, type II diabetes is the most common form and its prevalence varies by age, education, income, location, race, and other social determinants of health. Much of the burden of the disease falls on those of lower socioeconomic status as well. Diabetes also places a massive burden on the economy, with diagnosed diabetes costs of roughly $327 billion dollars and total costs with undiagnosed diabetes and prediabetes approaching $400 billion dollars annually.</div>', unsafe_allow_html=True)

st.write("""
***
""")
#How To Show Columns
columns = st.columns(3)
columns[0].button('Yes')
columns[1].button('No')
columns[2].button('Don know')

#importing The Complete DataSet into Streamlit
diabete_df=pd.read_csv("https://raw.githubusercontent.com/MehrdadGh1989/Diabete_Project/main/diabetes_012_health_indicators_BRFSS2015.csv")
#Clean the DataFrame from duplicated rows
diabete_df_clean=diabete_df.drop_duplicates()
diabete_df_clean.reset_index(drop=True,inplace=True)
diabete_df_clean
####################################################
#Some Changes in Data Frame

####################################################
#How to use different columns
#col_1,col_2=st.columns(2)
#st.subheader('Plots')
#with col_1:
#    fig1=NoBinaryData_df['Disease Status'].value_counts().plot.bar()
#    st.write(fig1)




#with col_2:

#Heatmap Corr
#=plt.figure(figsize=(20,15))
#sns.heatmap(diabete_df.corr(), annot=True)
#st.write(A)