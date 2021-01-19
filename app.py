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

	st.title("BMI Streamlit App")
	#st.subheader("Anaylysis and BMI Prediction")


	   
	    


	    #reading the data
	st.cache()
	data=pd.read_csv('bmi_data.csv')
	

	task=['Home','Data Analysis','BMI Prediction']
	st.sidebar.header("BMI Application")
	tasksel=st.sidebar.selectbox("Select Task",task)
	if tasksel=='Home':
		st.image('bmi.jpg',width=700)



	if tasksel=='Data Anaylysis':
		if st.button("Preview BMI Data"):
			st.write(data)
 
		st.subheader("Exploring The Dataset")
		if st.button("Describe Data"):
			describtion=data.describe()
			st.write(describtion)

		if st.button("Check Null Values"):
		    na=data.isna().sum(axis=0)
		    st.write(na)

		if st.button("Clean Data"):
		    data.Height.fillna(data.Height.mean(),inplace=  True)
		    data.Weight.fillna(data.Weight.mean(),inplace=  True)
		    data.BMI.fillna(data.BMI.mean(),inplace=  True)
		    st.write(data.isna().sum(axis=0))


		if st.button("Check For Data Correlation"):
			st.subheader("Identify Correlation")
			sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
			st.pyplot()

		st.subheader("Lets Do some Visualization")
		if st.checkbox("Show Scatter BMI vs Weight"):
		    sns.lmplot(x='Weight',y='BMI',data=data)
		    st.pyplot()
		    st.markdown("From observation we can see than an Increase in Weight affects the BMI")
		if st.checkbox("Show Scatter BMI vs Age"):
		    sns.lmplot(x='Age',y='BMI',data=data)
		    st.pyplot()
		    st.markdown("From observation we can see that Age doesnt have any effect on a persons BMI")
		if st.checkbox("Show Scatter BMI vs Height"):
		    sns.lmplot(x='Height',y='BMI',data=data)
		    st.pyplot()
		    st.markdown("From observation we can see that Height tends to have an effect on a persons BMI")
		           
		      
		if st.checkbox("Show Gender Count "):
		    sns.countplot(x='Sex',data=data)
		    st.pyplot()

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

		if st.button("Build and Check Model Score"):
			st.write("The model Prediction Accuracy is :",result,"%")

	if tasksel=='BMI Prediction':
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
