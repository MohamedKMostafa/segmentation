# Importing necessary libraries
import numpy as np
import pandas as pd 
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans

# Reading the data
RFM = pd.read_csv(r'D:\courses\Technu lab\machine learning final project\Customer Segmentation\online+retail+ii\RFM Backup.csv')
RFM_model=RFM.copy()
RFM_model.drop(columns=['Customer ID', 'Country','Segmentation',"R Score","F Score","M Score","RFM Score"],inplace=True)



categorical_cols = ['Country', 'Segmentation']
categorical_data = RFM[categorical_cols]
le = LabelEncoder()
RFM[categorical_cols] = categorical_data.apply(lambda col: le.fit_transform(col))



sc=StandardScaler()
sc.fit_transform(RFM)
# Training the model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit_predict(RFM)
model=KMeans()


#with open('model.pkl', 'wb') as file:
    #pickle.dump(model,"RFM")

path = 'D:\\courses\\Technu lab\\machine learning final project\\Customer Segmentation\\online+retail+ii\\model.pkl'
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))
pickle.dump(le, open('encoder.pkl', 'wb'))



model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))
