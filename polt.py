import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.classification import *


data=pd.read_csv("sample.csv",encoding='UTF-8')

print(data)

plt.figure(figsize=(15, 10))
sns.countplot(data.Potability)
plt.title("Distribution of Unsafe and Safe Water")
plt.show()

data = data.dropna()
data.isnull().sum()

data=data
figure = px.histogram(data, x = "ph", color = "Potability", title= "Factors Affecting Water Quality: PH")
figure.show()

figure=px.histogram(data,x="Hardness",color="Potability",title="Factors Affecting water Quality:Hardness")
figure.show()

figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
figure.show()

figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
figure.show()

figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()

figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()

figure = px.histogram(data, x = "Organic_carbon", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()

figure = px.histogram(data, x = "Trihalomethanes", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()

figure = px.histogram(data, x = "Turbidity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Turbidity")
figure.show()

'''----'''
correlation = data.corr()
print(correlation["ph"].sort_values(ascending=False))

'''----'''

data = data.dropna(subset=['Potability'])


clf = setup(data, target="Potability", verbose=False, session_id=786)

compare_models()

'''----'''

model = create_model("rf")
predict = predict_model(model, data=data)
print(predict.head())

'''----'''
model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()

