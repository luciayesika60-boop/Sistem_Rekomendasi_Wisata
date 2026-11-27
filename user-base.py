import pandas as pd
import numpy as np

Rating="D:/Sisrek_Wisata/Dataset_Rating_Mgl (2).csv"
Tempat="D:/Sisrek_Wisata/Dataset_tourisMagelang (3).csv"
User ="D:/Sisrek_Wisata/Dataset_usermgl (3).csv"

def baca_data():
    rating=pd.read_csv(Rating)
    matrix=rating.pivot_table(index="User_Id",columns="Place_Name",values="Place_Rating")
    return rating,matrix

def Userbase_sim(matrix,u,i):
    user_rating = matrix.loc[[u,i]].dropna(axis=1)
    if user_rating.shape[1]<2:
        return 0.0
    a,b = user_rating.loc[u], user_rating[i]
    a= a- matrix.loc[u].mean()
    b= b- matrix.loc[i].mean()
    atas=(a*b).sum()
    bawah=np.sqrt((a*a).sum())*np.sqrt((b*b).sum())
    if bawah == 0 :
        return 0.0
    else: 
        hasil= atas/bawah
        return hasil
    
if __name__=="__main__":
    rating,matrix= baca_data()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

def prediksi(matrix)

    

 





    


