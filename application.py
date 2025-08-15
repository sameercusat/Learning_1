import streamlit as st
import pandas as pd
from src.utils import load_object
import os

st.title("Welcome to Prediction Page")

gender_map={'Male':'male','Female':'female'}

gender_face=st.selectbox('Choose Gender:',list(gender_map.keys()))

gender=gender_map[gender_face]

race_map={'Group A':'group A','Group B':'group B','Group C':'group C','Group D':'group D','Group E':'group E'}

race_face=st.selectbox('Choose Race or Ethinicity:',list(race_map.keys()))

race=race_map[race_face]

parent_edu_map={'Some College':'some college','Associate Degree':"associate's degree",'High School':'high school','Some High School':'some high school','Bachelor Degree':"bachelor's degree",'Master Degree':"master's degree"}

parent_edu_face=st.selectbox('Choose Parent Level of Education:',list(parent_edu_map.keys()))

parent_edu=parent_edu_map[parent_edu_face]

lunch_map={'Free or Reduced':'free/reduced','Standard':'standard'}

lunch_face=st.selectbox('Choose Lunch:',list(lunch_map.keys()))

lunch=lunch_map[lunch_face]

course_map={'None':'none','Completed':'completed'}

course_face=st.selectbox('Choose Test Prepration Course:',list(course_map.keys()))

course=course_map[course_face]

writing_score=st.slider('Select Writing Score:',0,100,0)

reading_score=st.slider('Select Reading Score:',0,100,0)


data={'gender':gender,'race_ethnicity':[race],'parental_level_of_education':[parent_edu],'lunch':[lunch],'test_preparation_course':[course],'writing_score':[writing_score],'reading_score':[reading_score]}
dataframe=pd.DataFrame(data)

#st.write(dataframe)
preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
preprocessor=load_object(preprocessor_path)
scaled_data=preprocessor.transform(dataframe)
#print(scaled_data)
model_path=os.path.join('artifacts','model_trainer.pkl')
model=load_object(model_path)
preds=model.predict(scaled_data)

st.write("Predicted Math Score is:")
st.write(preds)



