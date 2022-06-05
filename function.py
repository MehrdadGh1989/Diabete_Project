#Creating table which contains columns with Binary Data and non-binary
list_with_binary_attributes=['Diabetes_012','Veggies','HighBP','HighChol','CholCheck','Sex','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','HighChol','DiffWalk','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost']
BinaryData_df=diabete_df_clean[list_with_binary_attributes]
list_with_no_binary_attributes=['Diabetes_012','BMI','PhysHlth','MentHlth','Age','Income','Education','GenHlth']
NoBinaryData_df=diabete_df_clean[list_with_no_binary_attributes]
#NOW THE AIM HERE IS TO CATEGORIZE (Not Binary Datas) In a way which will be more undestandable 
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