from dataclasses import asdict
from turtle import width
from unittest.util import sorted_list_difference
import streamlit as st
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
from PIL import Image
import plotly.figure_factory as ff
import altair as alt
#import function
#How to run Streamlit
#cd C:\Users\ghadi\source\repos\Presentation_Diabete_Project_Streamlit\Presentation_Diabete_Project_Streamlit
# streamlit run Presentation_Diabete_Project_Streamlit.py

#DataSet And Cleaning and Clustering to different DataSets(With BinaryData-Categorical Data)
#Importing DataSet
diabete_df=pd.read_csv("https://raw.githubusercontent.com/MehrdadGh1989/Diabete_Project/main/diabetes_012_health_indicators_BRFSS2015.csv")
#Clean the DataFrame from duplicated rows
diabete_df_clean=diabete_df.drop_duplicates()
diabete_df_clean.reset_index(drop=True,inplace=True)

#Creating DataFrame which contains columns with Binary Data and Categorical Data
list_with_binary_attributes=['Diabetes_012','Veggies','HighBP','CholCheck', 'Sex', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'HighChol', 'DiffWalk', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost']
BinaryData_df=diabete_df_clean[list_with_binary_attributes]
list_with_no_binary_attributes=['Diabetes_012','BMI','PhysHlth','MentHlth','Age','Income','Education','GenHlth']
NoBinaryData_df=diabete_df_clean[list_with_no_binary_attributes]

#NOW THE AIM HERE IS TO CATEGORIZE (Not Binary Datas=Categorical DATAS) In a way which will be more undestandable 
# (Groping Data- Organizing data) in specified Categories
#1 Obse         BMI >= 30
#2 Overweight   BMI   25-29.9
#3 Healthy      BMI   18.5-24.9
#4 UnderWeight  BMT < 18.5
diabete_weight_group=NoBinaryData_df['BMI']
NoBinaryData_df.loc[diabete_weight_group <18.5, 'Weight_Group']='Under Weight'
NoBinaryData_df.loc[((diabete_weight_group >= 18.5) & (diabete_weight_group <= 24.9)), 'Weight_Group']='Healthy'
NoBinaryData_df.loc[((diabete_weight_group >= 25) & (diabete_weight_group <= 29.9)), 'Weight_Group']='Over Weight'
NoBinaryData_df.loc[diabete_weight_group >=30, 'Weight_Group']='Obse'
###########
#Age_group_detection
#1   Age 18-24
#2   Age 25-29
#3   Age 30-34
#4   Age 35-39
#5   Age 40-44
#6   Age 45-49
#7   Age 50-54
#8   Age 55-59
#9   Age 60-64
#10  Age 65-69
#11  Age 70-74
#12  Age 75_79
#13  Age 80 or older
diabete_age_group=NoBinaryData_df['Age']
NoBinaryData_df.loc[diabete_age_group==1,'Age_Group']='Age 18-24'
NoBinaryData_df.loc[diabete_age_group==2,'Age_Group']='Age 25-29'
NoBinaryData_df.loc[diabete_age_group==3,'Age_Group']='Age 30-34'
NoBinaryData_df.loc[diabete_age_group==4,'Age_Group']='Age 35-39'
NoBinaryData_df.loc[diabete_age_group==5,'Age_Group']='Age 40-44'
NoBinaryData_df.loc[diabete_age_group==6,'Age_Group']='Age 45-49'
NoBinaryData_df.loc[diabete_age_group==7,'Age_Group']='Age 50-54'
NoBinaryData_df.loc[diabete_age_group==8,'Age_Group']='Age 55-59'
NoBinaryData_df.loc[diabete_age_group==9,'Age_Group']='Age 60-64'
NoBinaryData_df.loc[diabete_age_group==10,'Age_Group']='Age 65-69'
NoBinaryData_df.loc[diabete_age_group==11,'Age_Group']='Age 70-74'
NoBinaryData_df.loc[diabete_age_group==12,'Age_Group']='Age 75_79'
NoBinaryData_df.loc[diabete_age_group==13,'Age_Group']='Age 80 or older'
#Education_group_detection
#1 Not attended School
#2 Elementary
#3 Some High School
#4 High School Graduated
#5 Some College or Technichal School  
#6 College Graduate
diabete_education_group=NoBinaryData_df['Education']
NoBinaryData_df.loc[diabete_education_group==1,'Education_Level']='Not attended School'
NoBinaryData_df.loc[diabete_education_group==2,'Education_Level']='Elementary'
NoBinaryData_df.loc[diabete_education_group==3,'Education_Level']='Some High School'
NoBinaryData_df.loc[diabete_education_group==4,'Education_Level']='High School Graduated'
NoBinaryData_df.loc[diabete_education_group==5,'Education_Level']='Some College or Technichal School'
NoBinaryData_df.loc[diabete_education_group==6,'Education_Level']='College Graduate'

###################
#Income_level
#1=Less than 10000 $
#2=Less than 15000 $
#3=Less than 20000 $
#4=Less than 25000 $
#5=Less than 35000 $
#6=Less than 50000 $
#7=Less than 75000 $
#8=75000 $ or More
diabete_income_group=NoBinaryData_df['Income']
NoBinaryData_df.loc[diabete_income_group==1,'Income_Level']='Less than 10000 $'
NoBinaryData_df.loc[diabete_income_group==2,'Income_Level']='Less than 15000 $'
NoBinaryData_df.loc[diabete_income_group==3,'Income_Level']='Less than 20000 $'
NoBinaryData_df.loc[diabete_income_group==4,'Income_Level']='Less than 25000 $'
NoBinaryData_df.loc[diabete_income_group==5,'Income_Level']='Less than 35000 $'
NoBinaryData_df.loc[diabete_income_group==6,'Income_Level']='Less than 50000 $'
NoBinaryData_df.loc[diabete_income_group==7,'Income_Level']='Less than 75000 $'
NoBinaryData_df.loc[diabete_income_group==8,'Income_Level']='75000 $ or More'
#General Health:
#1 = excellent,
#2 = very good,
#3 = good,
#4 = fair,
#5 = poor
diabete_GeneralHealth_group=NoBinaryData_df['GenHlth']
NoBinaryData_df.loc[diabete_GeneralHealth_group==1,'GeneralH']='Excellent'
NoBinaryData_df.loc[diabete_GeneralHealth_group==2,'GeneralH']='Very Good'
NoBinaryData_df.loc[diabete_GeneralHealth_group==3,'GeneralH']='Good'
NoBinaryData_df.loc[diabete_GeneralHealth_group==4,'GeneralH']='Fair'
NoBinaryData_df.loc[diabete_GeneralHealth_group==5,'GeneralH']='Poor'
#Diabete Status
# No Diabete=0
# Pre Diabete=1
# Diabete = 2
diabete_type_detection=NoBinaryData_df['Diabetes_012']
NoBinaryData_df.loc[diabete_type_detection==0,'Disease Status']='No Diabete'
NoBinaryData_df.loc[diabete_type_detection==1,'Disease Status']='Pre Diabete'
NoBinaryData_df.loc[diabete_type_detection==2,'Disease Status']='Diabete'
######
diabete_type_detection=BinaryData_df['Diabetes_012']
BinaryData_df.loc[diabete_type_detection==0,'Disease Status']='No Diabete'
BinaryData_df.loc[diabete_type_detection==1,'Disease Status']='Pre Diabete'
BinaryData_df.loc[diabete_type_detection==2,'Disease Status']='Diabete'
#################################################################
###### Codes Related To Data Visualisation ######################
#################################################################
#1-The General Status of Pepole (Observations)
NoBinaryData_df['Disease Status'].value_counts()
#2-Showing the Data related to General Health In Bar Chart
NoBinaryData_df['Disease Status'].value_counts().plot.bar()


#################################################################
    
#image=Image.open('C:/Users/ghadi/Desktop/Univr2.jpg')
#st.image(image,use_column_width=True)
#Splitting the part in streamlit
#st.write("""
#***
#""")
#Choosing Title in streamlit
st.title('Programming Project')

st.write("""
***
""")
#######################################
if st.sidebar.checkbox('Data Part'):
    if st.checkbox('A Concise Explanation about the Project'):
        st.write('Explanation About the Project')
        # How to Visualize (Justify)-TextFiles According to WordShape
        Diabete_Explanation=st.markdown('<div style="text-align: justify;">Diabetes is among the most prevalent chronic diseases in the United States, impacting millions of Americans each year and exerting a significant financial burden on the economy. Diabetes is a serious chronic disease in which individuals lose the ability to effectively regulate levels of glucose in the blood, and can lead to reduced quality of life and life expectancy. After different foods are broken down into sugars during digestion, the sugars are then released into the bloodstream. This signals the pancreas to release insulin. Insulin helps enable cells within the body to use those sugars in the bloodstream for energy.Diabetes is generally characterized by either the body not making enough insulin or being unable to use the insulin that is made as effectively as needed.\n Complications like heart disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high levels of sugar remaining in the bloodstream for those with diabetes. While there is no cure for diabetes, strategies like losing weight, eating healthily, being active, and receiving medical treatments can mitigate the harms of this disease in many patients. Early diagnosis can lead to lifestyle changes and more effective treatment, making predictive models for diabetes risk important tools for public and public health officials.\nThe scale of this problem is also important to recognize. The Centers for Disease Control and Prevention has indicated that as of 2018, 34.2 million Americans have diabetes and 88 million have prediabetes. Furthermore, the CDC estimates that 1 in 5 diabetics, and roughly 8 in 10 prediabetics are unaware of their risk. While there are different types of diabetes, type II diabetes is the most common form and its prevalence varies by age, education, income, location, race, and other social determinants of health. Much of the burden of the disease falls on those of lower socioeconomic status as well. Diabetes also places a massive burden on the economy, with diagnosed diabetes costs of roughly $327 billion dollars and total costs with undiagnosed diabetes and prediabetes approaching $400 billion dollars annually.\n The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey that is collected annually by the CDC. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. For this project, a csv of the dataset available on Kaggle for the year 2015 was used. This original dataset contains responses from 441,455 individuals and has 330 features. These features are either questions directly asked of participants, or calculated variables based on individual participant responses.</div>', unsafe_allow_html=True)
    st.write("""
    ***
    """)
    if st.checkbox('Show Different DataFrames'):
        st.write('There are 4 DataFrames, Which One would You like to see?')
        if st.checkbox('The Main DataFrame Released in The Kaggle Website'):
            st.write(diabete_df)
        if st.checkbox('The Clean DataFrame: Deleting The Duplicated rows'):
            st.write('diabete_df_clean')
        if st.checkbox('The Binary DataFrame Extracted from The Clean DataFrame'):
            st.write(BinaryData_df)
        if st.checkbox('The Categorical DataFrame Extracted from The Clean DataFrame'):
            st.write(NoBinaryData_df)
    st.write("""
    ***
    """)
st.write("""
***
""")

####################################################
#Data Visualization Part
####################################################
#Sidebar
if st.sidebar.checkbox('Data Visualization Part'):
    st.header('Data Visualization Part')
    if st.checkbox('Education Level of People participated in this Survey'):
        Education_Level=NoBinaryData_df['Education_Level'].value_counts()
        #(Education-Level in Percentage)
        colors = ['darkorange', 'sandybrown','darksalmon', 'orangered','chocolate']
        #st.write(Education_Level)
        #PieChart - Education Level
        fig1, ax1 = plt.subplots(figsize=(10,6))
        labels=['Not attended School','Elementary','Some High School','High School Graduated','Some College or Technichal School','College Graduate']
        sizes=[174,4040,9467,61158,66499,88443]
        ax1.pie(sizes, explode=[0.1,0.5,0.1,0.1,0.1,0.1], labels=labels, autopct='%1.1f%%',colors=colors,shadow=False)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
        st.caption('The Percentage of People in different Education Levels')
        
        Education_Level=NoBinaryData_df['Education_Level'].value_counts()
       # st.write(Education_Level)
        #with col_2:
        st.write("""
        ***
        """)
    ########################################################################
        #2 (Disease Status in pie Chart in percentage)
    if st.checkbox('Diabete Status of People participated in this Survey'):
        fig2,ax2=plt.subplots(figsize=(10,6))
        colors = ['cornflowerblue','silver', 'lime']
        #Disease_Status=NoBinaryData_df['Disease Status'].value_counts()
        #st.write(Disease_Status)
        #PieChart - Disease Status
        labels=['No Diabete','Diabete','Pre diabete']
        sizes=[190055,35097,4629]
        ax2.pie(sizes, explode=[0.0,0.0,0.0], labels=labels, autopct='%1.1f%%',colors=colors,shadow=False)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig2)
        st.caption('The Percentage of people Having No Diabete, Daibete or Pre Diabete')
        st.write("""
        ***
        """)
    ####################################################################
    #3Comparing The General Health condition of Smokers and NonSmokers
    if st.checkbox('Smoke and General Health'): 
        fig3,ax3=plt.subplots(figsize=(10,6))
        df = pd.DataFrame({'Excellent': {'NonSmoker': 21902,'Smoker':13005},
                   'VeryGood' : {'NonSmoker': 44203,'Smoker':33333},
                    'Good'    : {'NonSmoker': 38350,'Smoker':35364},
                    'Fair'    : {'NonSmoker': 14088,'Smoker':17458},     
                    'Poor'    : {'NonSmoker': 4238 ,'Smoker': 7840}})
                  
        sns.barplot(x='General Health', y='value', hue='index',
               data=df.reset_index().melt(id_vars='index', var_name='General Health'))
        st.write(fig3)
        st.caption('Comparing The General Health condition of Smokers and NonSmokers')
        st.write("""
        ***
        """)
#######################################
    #4 Heat Map (Correlation-Heatmap)
    if  st.checkbox('Correlation-HeatMap'):
        A=plt.figure(figsize=(20,15))
        sns.heatmap(diabete_df_clean.corr(), annot=True,cmap = 'YlOrBr')
        st.write(A)
        st.caption('Correlation between different independent variables')
        st.write("""
        ***
        """)
##########################################################
    #5 Density Distribution 
    
    if  st.checkbox('Density-Distribution'):
        col3,col4=st.columns(2)
        with col3:
            fig4,ax4=plt.subplots(figsize=(10,6))
            sns.distplot(NoBinaryData_df["BMI"], hist_kws = dict(linewidth = 1,facecolor='peru',edgecolor = "g"),bins = 70);
            st.write(fig4)
            st.caption('Density Distribution of BMI Parameter')
            st.write("""
            ***
            """)
            #to check the distribution of GenHlth
            #the distribution like below since thay have (specified and limited number) which repeated among different
        with col4:
            fig5,ax5=plt.subplots(figsize=(10,6))
            sns.kdeplot(NoBinaryData_df['GenHlth'], shade=True)
            st.write(fig5)
            st.caption('Density Distribution of GenHlth Parameter')
            st.write("""
            ***
            """)
 ##########################################################################################
#Comparing Men and Women at different age groups Having Diabetes
    if st.checkbox('Comparing Men and Women Having diabete at different Age level'): 
        fig6,ax6=plt.subplots(figsize=(10,6))
        df1 = pd.DataFrame({'Age 18-24': {'Female': 45,'Male':33},
                   'Age 25-29' : {'Female': 89,'Male':51},
                   'Age 30-34'    : {'Female': 191,'Male':123},
                   'Age 35-39'    : {'Female': 366,'Male':259},     
                   'Age 40-44'    : {'Female': 576,'Male':473},
                   'Age 45-49'    : {'Female': 902,'Male':839},
                   'Age 50-54'    : {'Female': 1639,'Male':1433},
                   'Age 55-59'    : {'Female': 2250,'Male':1991},
                   'Age 60-64'    : {'Female': 2955,'Male':2726},
                   'Age 65-69'    : {'Female': 3220,'Male':3263},
                   'Age 70-74'    : {'Female': 2540,'Male':2550},
                   'Age 75_79'    : {'Female': 1832,'Male':1551},
                   'Age 80 or older'    : {'Female': 1740,'Male':1460}})
               
        sns.barplot(x='Age Level', y='value', hue='index',
               data=df1.reset_index().melt(id_vars='index', var_name='Age Level'))
        plt.xticks(label='ddm', rotation=25)
        st.write(fig6)
        st.caption('Comparing The Females and Males Having Diabete in Different Age Level ')
        st.write("""
        ***
        """)
#######################################
    #4 Heat Map (Correlation-Heatmap)
    if  st.checkbox('Histogram'):
        A=plt.figure(figsize=(20,15))
        diabete_df_clean.hist(bins=50,figsize=(30,40))
        st.write(A)
        st.caption('Histogram')
        st.write("""
        ***
        """)        

#1   Age 18-24
#2   Age 25-29
#3   Age 30-34
#4   Age 35-39
#5   Age 40-44
#6   Age 45-49
#7   Age 50-54
#8   Age 55-59
#9   Age 60-64
#10  Age 65-69
#11  Age 70-74
#12  Age 75_79
#13  Age 80 or older
##################################################################
######## Modifying Approach1
#Creating DATA FRAME ACCORDING TO OBSERVATION OF PREDIABETES
        #Second Approach
        #Having the same number of observations in each category and analyze it
        #
        #Creating New DataFrame Containing Only Observations with Pre Diabetes
Prediabete_df=diabete_df_clean[diabete_df_clean['Diabetes_012']==1]
Prediabete_df['Diabetes_012'].value_counts()
#Since it has Only 4629 rows, we want to contain these Observations and create other data frames based on these

#Creating New DataFrame Containing Only Observations with Diabetes
diabete_df_new=diabete_df_clean[diabete_df_clean['Diabetes_012']==2]
diabete_df_new['Diabetes_012'].value_counts()
#diabete_df_new
#Random Selecting of the above DataFrame (diabete_df)
diabete_newsample_df = diabete_df_new.sample(n=4629,replace=False)
diabete_newsample_df['Diabetes_012'].value_counts()
#diabete_newsample_df
#Creating New DataFrame Containing Only Observations with Non Diabetes
Nondiabete_df=diabete_df_clean[diabete_df_clean['Diabetes_012']==0]
Nondiabete_df['Diabetes_012'].value_counts()
#Random Selecting of the above DataFrame (Nondiabete_df)
Nondiabete_newsample_df = Nondiabete_df.sample(n=4629,replace=False)
Nondiabete_newsample_df['Diabetes_012'].value_counts()
#Nondiabete_newsample_df

#combined_equal_df=pd.merge(Prediabete_df,Nondiabete_newsample_df,diabete_newsample_df,on='Diabete_012')
combined_equal_df = pd.concat([Prediabete_df,diabete_newsample_df,Nondiabete_newsample_df], axis=0, join='inner')
combined_equal_df.describe().T.round(2)
combined_equal_df = combined_equal_df.sample(frac=1).reset_index(drop=True)
#st.write(combined_equal_df)
#colors=['darksalmon', 'orangered','chocolate']
#combined_equal_df['Disease Status'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',colors=colors,shadow=False,figsize=  (10,8))
#plt.show()
###############################################################################
#Modifying Approach 2-
#Creating a DataFrame which has only Diabete and NonDiabete Observations
#Removing PreDiabetes from dataframe (Having only Diabete and NoDiabete) and evaluating the Accuracy prediction
NonPreDiabete_df=diabete_df_clean.drop(diabete_df_clean.index[diabete_df_clean['Diabetes_012'] == 1])
#NonPreDiabete_df

##################################################################
#Machine Learning
#First Approach
##################################################################

if st.sidebar.checkbox('Machine Learning'):
    st.header('Prediction Part')
    st.write("""
        ***
        """)
    st.write("The main Target is Predicting Diabete Disease. At First, the Clean DataFrame has been selected to be investigated With 3 Different Machine Leaning Models.")
    st.write('The Below Figure extracted from Clean DataFrame shows the Percentage of Observations who have Diabte, PreDiabete and NonDiabete.')
    fig2,ax2=plt.subplots(figsize=(10,6))
    colors = ['slateblue','violet', 'slategrey']
    #Disease_Status=NoBinaryData_df['Disease Status'].value_counts()
    #st.write(Disease_Status)
    #PieChart - Disease Status
    labels=['No Diabete','Diabete','Pre diabete']
    sizes=[190055,35097,4629]
    ax2.pie(sizes, explode=[0.0,0.0,0.0], labels=labels, autopct='%1.1f%%',colors=colors,shadow=False)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig2)
    
    #######################333
    #Model 1 : Naive-Bayse
    if st.checkbox('Model 1: Naive-Bayes'):
        from sklearn import metrics
        from sklearn import metrics
        from sklearn.model_selection import train_test_split  
        from sklearn.preprocessing import StandardScaler 
        from sklearn.metrics import confusion_matrix 
 
        x = diabete_df_clean.iloc[:, [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values  
        y = diabete_df_clean.iloc[:, 0].values  

        x1 = diabete_df_clean.iloc[:, [1]].values  
        x2 = diabete_df_clean.iloc[:, [2]].values
        x3 = diabete_df_clean.iloc[:, [3]].values
        x4 = diabete_df_clean.iloc[:, [4]].values
        x5 = diabete_df_clean.iloc[:, [5]].values
        x6 = diabete_df_clean.iloc[:, [6]].values
        x7 = diabete_df_clean.iloc[:, [7]].values
        x8 = diabete_df_clean.iloc[:, [8]].values
        x9 = diabete_df_clean.iloc[:, [9]].values
        x10 = diabete_df_clean.iloc[:, [10]].values
        x11 = diabete_df_clean.iloc[:, [11]].values
        x12 = diabete_df_clean.iloc[:, [12]].values
        x13 = diabete_df_clean.iloc[:, [13]].values
        x14 = diabete_df_clean.iloc[:, [14]].values
        x15 = diabete_df_clean.iloc[:, [15]].values
        x16 = diabete_df_clean.iloc[:, [16]].values
        x17 = diabete_df_clean.iloc[:, [17]].values
        x19 = diabete_df_clean.iloc[:, [19]].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)  
        from sklearn.naive_bayes import GaussianNB  
        classifier = GaussianNB()  
        classifier.fit(x_train, y_train)  
        y_pred = classifier.predict(x_test)
        y_pred = classifier.predict(x_test) 
        print(y_pred)
        cm = confusion_matrix(y_test, y_pred)  
        print(cm)
        st.write('The Confusion Matrix is:')
        st.write(cm)
        st.write('y_Pridication,x_Actual')
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        st.write("Accuracy in Percentage:",metrics.accuracy_score(y_test, y_pred)*100)
        st.write("""
        ***
        """)
   #############
   #Decision Tree Model2
    if st.checkbox('Model 2: Decision Tree '):
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split 
        from sklearn import metrics 
        from sklearn.metrics import confusion_matrix 
        x = diabete_df_clean.iloc[:, [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values  
        y = diabete_df_clean.iloc[:, 0].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        abc = DecisionTreeClassifier(random_state=2)
        model = abc.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)  
        print(cm)
        st.write('The Confusion Matrix is:')
        st.write(cm)
        st.write('y_Pridication,x_Actual')
        st.write("Accuracy in Percentage:",metrics.accuracy_score(y_test, y_pred)*100)
        st.write("""
        ***
        """)
   #############
   #Adaboosting Model 3
    if st.checkbox('Model 3: Adaboosting '):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import train_test_split 
        from sklearn import metrics 
        from sklearn.metrics import confusion_matrix
        x = diabete_df_clean.iloc[:, [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values  
        y = diabete_df_clean.iloc[:, 0].values 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        Adaboost_Classifier = AdaBoostClassifier(n_estimators=70,learning_rate=1)
        model = Adaboost_Classifier.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100, "%")
        cm = confusion_matrix(y_test, y_pred)  
        print(cm)
        st.write('The Confusion Matrix is:')
        st.write(cm)
        st.write('y_Pridication,x_Actual')
        st.write("Accuracy in Percentage:",metrics.accuracy_score(y_test, y_pred)*100)
        st.write("""
        ***
        """)
  ########################################################################3
    if st.checkbox('Modifying Approach 1'):
        st.write('In this Approach the New DataFrame has been defined based on equal Observation. Also, In this approach, only the Adaboosting Model has been used since it had more accuracy in comparison to other models.')
        st.write(combined_equal_df)
        fig9,ax9=plt.subplots(figsize=(10,6))
        colors = ['slateblue','violet', 'slategrey']
        #Disease_Status=NoBinaryData_df['Disease Status'].value_counts()
        #st.write(Disease_Status)
        #PieChart - Disease Status
        labels=['No Diabete','Diabete','Pre diabete']
        sizes=[4629,4629,4629]
        ax9.pie(sizes, explode=[0.0,0.0,0.0], labels=labels, autopct='%1.1f%%',colors=colors,shadow=False)
        ax9.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig9)
        st.caption('The pie chart shows the percentage of Observations with different Health Status')
        #Use adaboosting for new data-frame
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        x = combined_equal_df.iloc[:, [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values  
        y = combined_equal_df.iloc[:, 0].values 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        Adaboost_Classifier = AdaBoostClassifier(n_estimators=70,learning_rate=1)
        model = Adaboost_Classifier.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.write("Accuracy in Percentage:",metrics.accuracy_score(y_test, y_pred)*100)
 ################################################################################################################   
    if st.checkbox('Modifying Approach 2'):
       st.write('In this Approach the New DataFrame has been defined based on NonDiabete and Diabete Observations (PreDiabete has been removed from DataFrame). Also, In this approach, only the Adaboosting Model has been used since it had more accuracy in comparison to other models.')
       
       st.write(combined_equal_df)
       fig10,ax10=plt.subplots(figsize=(10,6))
       colors = ['slateblue','violet']
       labels=['No Diabete','Diabete']
       sizes=[190055,35097]
       ax10.pie(sizes, explode=[0.0,0.0], labels=labels, autopct='%1.1f%%',colors=colors,shadow=False)
       ax10.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
       st.pyplot(fig10)
       st.caption('The pie chart shows the percentage of Observations with different Health Status')

       from sklearn.ensemble import AdaBoostClassifier
       from sklearn.model_selection import train_test_split
       from sklearn import metrics
       x = NonPreDiabete_df.iloc[:, [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values  
       y = NonPreDiabete_df.iloc[:, 0].values 
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
       Adaboost_Classifier = AdaBoostClassifier(n_estimators=70,learning_rate=1)
       model = Adaboost_Classifier.fit(x_train, y_train)
       y_pred = model.predict(x_test)
       st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100, "%")

###############
