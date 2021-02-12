import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
	st.set_option('deprecation.showPyplotGlobalUse', False)

	#st.title("BMI Streamlit App")
	#st.subheader("Anaylysis and BMI Prediction")






	   
	    
 
	    #reading the data
	@st.cache()
	def loaddata():
		data=pd.read_csv('bmi_data.csv')
		data.Height.fillna(data.Height.mean(),inplace=  True)
		data.Weight.fillna(data.Weight.mean(),inplace=  True)
		data.BMI.fillna(data.BMI.mean(),inplace=  True)
		return data
	data=loaddata()
	 

	Menu=['About','Dashboard','BMI Prediction']
	st.sidebar.header("BMI Prediction Application")
	Menusel=st.sidebar.selectbox("Select Menu Option",Menu)
	if Menusel=='About':
		st.title("About")
		header_html="""<hr>"""
		st.markdown(header_html, unsafe_allow_html=True,)

		st.info("The World Health Organization will like every one to know and identify the factors that affect ones Body Mass Index.")
		st.info('Body Mass Index (BMI) is a person’s weight in kilograms divided by the square of height in meters. A high BMI can be an indicator of high body fatness. BMI can be used to screen for weight categories that may lead to health problems but it is not diagnostic of the body fatness or health of an individual.')
		st.info('f your BMI is between 18.5-24.9: Your BMI is considered normal. This healthy weight helps reduce your risk of serious health conditions and means you are close to your fitness goals. If your BMI is between 25-29.9: Your BMI is considered overweight. Being overweight may increase your risk of cardiovascular disease')
		



	if Menusel=='Dashboard':
		st.title("Dashboard")
		header_html="""<hr>"""
		st.markdown(header_html,unsafe_allow_html=True,)
		col1, col2 = st.beta_columns(2)
		with col1:
			col1.subheader("Dataset")
			col1.write(data.head(8))
		with col2:
			col2.subheader("Data Describtion")
			col2.write(data.describe())
		
		st.subheader("Data Visualization")
		st.info("Visualizing our dataset ")

		r2col1, r2col2=st.beta_columns(2)
		with r2col1:
			r2col1.subheader("Data Correlation")
			sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
			st.pyplot()
		with r2col2:
			r2col2.subheader("BMI Vs Age")
			sns.lmplot(x='Age',y='BMI',data=data)
			st.pyplot()

		st.info("Inference: Weight and Height are the main factors that affects ones BMI")

		r3col1, r3col2=st.beta_columns(2)
		
		with r3col1:
			r3col1.subheader("Weight Vs BMI")
			sns.lmplot(x='Weight',y='BMI',data=data)
			st.pyplot()
		with r3col2:
			r3col2.subheader("Height Vs BMI")
			sns.lmplot(x='Height',y='BMI',data=data)
			st.pyplot()
		st.info("Inference: Increase in Height decreases ones BMI value")


		x=data.drop(columns=['Sex','Age','BMI'])
		y=data['BMI'] 
		xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=0.8,test_size=0.2,)
		lmodel=LinearRegression()
		lmodel.fit(xtrain,ytrain)
		a=lmodel.score(xtest.values,ytest.values)
		result=a*100
 
		if st.button("Build and Check Model Score"):
			st.write("The model Prediction Accuracy is :",result,"%")

	if Menusel=='BMI Prediction':
		st.subheader('Select Input Parameters')
		def user_input_features():
			Height = st.slider('Height(Inches)', 60.0, 80.0, 60.0)
			Weight = st.slider('Weight(Pounds)', 70.0, 180.0, 70.0)
			dataset = {'Height(Inches)': Height,'Weight(Pounds)': Weight}
			features= pd.DataFrame(dataset,index=[0])
			return features
		df=user_input_features()
		st.write(df)

		data.Height.fillna(data.Height.mean(),inplace=  True)
		data.Weight.fillna(data.Weight.mean(),inplace=  True)
		data.BMI.fillna(data.BMI.mean(),inplace=  True)
		x=data.drop(columns=['Sex','Age','BMI'])
		y=data['BMI']
		xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=0.8,test_size=0.2,)
		lmodel=LinearRegression()
		lmodel.fit(xtrain,ytrain)
		a=lmodel.score(xtest.values,ytest.values)
		result=a*100
		BMIPrediction=lmodel.predict(df)

		st.write("Your BMI is :",BMIPrediction)
		if BMIPrediction < 18.5:
			st.warning("You are underweight")
		elif BMIPrediction >=18.5 and BMIPrediction <= 25.0:
			st.success("You are of Normal BMI")
		elif BMIPrediction >25.0 and BMIPrediction < 30.0:
			st.success("You are Over Weight")
		elif BMIPrediction >= 30:
			st.warning("You are Obese")
		else:
			st.info("Enter The Right Details")

 
	   

    
if __name__ == '__main__':
    main()
