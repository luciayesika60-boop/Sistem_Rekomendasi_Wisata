import pandas as pd
import numpy as np

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

df = pd.read_csv("archive (7)/DATASET/Dataset_Rating_Mgl.csv")
df_matrix = df.pivot_table(index="User_Id", columns="Place_Name", values="Place_Rating")
place = df_matrix.columns

user = int(input("Masukkan ID user: "))
user_ada = user in df_matrix.index

# Hanya sementara nanti disesuaikan dengan gui!
tempat_suka = []
for i in range(2):
    tempat = input("Masukkan nama tempat yang kamu suka: ")
    if tempat not in place:
        raise SystemExit("Tempat tidak ada di dataset: " + tempat)
    tempat_suka.append(tempat)

# Hitung cosine similarity antar tempat (kolomnya)
def cosine_similarity(df_matrix, i, j):
    i_j = df_matrix[[i, j]].dropna()
    if i_j.empty:
        return 0.0

    a = i_j[i].values
    b = i_j[j].values

    panjang_a = np.sqrt(np.sum(a ** 2))
    panjang_b = np.sqrt(np.sum(b ** 2))
    if panjang_a == 0 or panjang_b == 0:
        return 0.0

    return np.sum(a * b) / (panjang_a * panjang_b)

# Hitung rating dari 5 kolom yg mirip dengan yg dipilih
K = 5
hasil_rating = {}

for t in tempat_suka:
    sim = {}
    for j in place:
        if j != t:
            sim[j] = cosine_similarity(df_matrix, t, j)

    sim_t = pd.Series(sim)
    top5 = sim_t.sort_values(ascending=False).head(K)

    rating_user = pd.Series(np.nan, index=top5.index)
    rating_tetangga = rating_user.fillna(df_matrix[top5.index].mean()).dropna()

    sim_valid = top5[rating_tetangga.index]

    hasil_rating[t] = (sim_valid * rating_tetangga).sum() / sim_valid.sum()

#output hasil rating untuk yg dipilih user blum dalam bentuk data frame
user_baru = pd.Series(hasil_rating, name=user)


# for t, r in hasil_rating.items():
#     print(f"Rating {user} terhadap {t} = {round(r)}")
for i, r in user_baru.items():
    df_matrix_baru = pd.concat([df_matrix, user_baru.to_frame().T])
    df_matrix_baru.index.name = "User_Id"
    print(df_matrix_baru)