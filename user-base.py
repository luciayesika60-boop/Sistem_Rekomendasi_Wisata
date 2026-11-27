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
    a,b = user_rating.loc[u], user_rating.loc[i]
    a= a- matrix.loc[u].mean()
    b= b- matrix.loc[i].mean()
    atas=(a*b).sum()
    bawah=np.sqrt((a*a).sum())*np.sqrt((b*b).sum())
    if bawah == 0 :
        return 0.0
    else: 
        hasil= atas/bawah
        return hasil

def prediksi(matrix,x,tempat,k=5):
    mean_u =matrix.loc[x].mean()
    kandidat=[]
    for v in matrix.index:
        if v == x or pd.isna(matrix.loc[v,tempat]):
            continue
        mirip =Userbase_sim(matrix,x,v)
        if mirip > 0 :
            selisih=matrix.loc[v,tempat] - matrix.loc[v].mean()
            kandidat.append((mirip,selisih))
    kandidat =sorted(kandidat,key=lambda x:x[0],reverse=True)[:k]
    pembagi =sum(abs(mirip) for mirip, _ in kandidat)
    if pembagi == 0 :
        return mean_u
    pred=mean_u + sum(mirip * d for mirip, d in kandidat)/pembagi
    return min(max(pred,1),5)

def isi_prediksi (matrix,k=5):
    akhir = matrix.copy()
    for x in matrix.index:
        for tempat in matrix.columns[matrix.loc[x].isna()]:
            akhir.loc[x,tempat]=prediksi(matrix,x,tempat,k)
    return akhir

    
if __name__=="__main__":
    rating,matrix= baca_data()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    awal=matrix.iloc[:10, :5]
    #print data awal kosong
    print(awal)

    u = matrix.index[0]
    kosong = matrix.columns[matrix.loc[u].isna()]
    print(f"Prediksi user {u} untuk {kosong[0]}: {prediksi(matrix, u, kosong[0]):.2f}")

    # isi semua sel kosong
    matrix_penuh = isi_prediksi(matrix, k=5)
    print(matrix_penuh.iloc[:10, :5].round(0))


 





    


