import pandas as pd
import numpy as np

df = pd.read_csv("archive (7)/DATASET/Dataset_Rating_Mgl.csv")

df_matrix = df.pivot_table(index="User_Id", columns="Place_Name", values="Place_Rating")

user = int(input("Masukkan nama user: "))
data_user_new = []
for i in range(2):
    tempat1 = input("Masukkan nama tempat: ")
    toRating1 = int(input("Seberapa sering Anda mengunjungi tempat seperti pada gambar tersebut? "))
    data_user_new.append({"User_Id": user, "Place_Name": tempat1, "Place_Rating": toRating1})

data = pd.concat([df, pd.DataFrame(data_user_new)], ignore_index=True)
df_matrix_copy = data.pivot_table(index="User_Id", columns="Place_Name", values="Place_Rating")


def cosine_similarity(df_matrix_copy, i, j):
    i_j = df_matrix_copy[[i, j]].dropna()
    if i_j.empty:
        return 0.0

    a = i_j[i].values
    b = i_j[j].values

    dot_product = np.sum(a * b)
    panjang_a = np.sqrt(np.sum(a ** 2))
    panjang_b = np.sqrt(np.sum(b ** 2))

    if panjang_a == 0 or panjang_b == 0:
        return 0.0

    # similarity JANGAN dibulatkan, biarkan desimal penuh
    return dot_product / (panjang_a * panjang_b)


place = df_matrix_copy.columns
sim = pd.DataFrame(index=place, columns=place, dtype=float)

for i in place:
    for j in place:
        sim.loc[i, j] = cosine_similarity(df_matrix_copy, i, j)


def prediksi(df_matrix_copy, sim, user, k=5):
    rated = df_matrix_copy.loc[user].dropna()
    hasil = {}
    for item in df_matrix_copy.columns:
        if item in rated.index:
            continue
        s = sim.loc[item, rated.index].sort_values(ascending=False).head(k)
        s = s[s > 0]
        if s.sum() == 0:
            continue
        # hanya hasil akhir yang dibulatkan
        hasil[item] = int(round(np.sum(s.values * rated[s.index].values) / s.sum()))
    return pd.Series(hasil, dtype=int).sort_values(ascending=False)


pred = prediksi(df_matrix_copy, sim, user)

# Top 5 akhir dimasukkan ke dataframe baru
top5_akhir = pred.head(5).to_frame("Prediksi_Rating")
print(top5_akhir)

# Mengisi kekosongan rating user dengan hasil prediksi
df_matrix_isi = df_matrix_copy.copy()
df_matrix_isi.loc[user, pred.index] = pred
print(df_matrix_isi.loc[user].dropna().round().astype(int))