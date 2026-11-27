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


def cosine_similarity(df_matrix, i, j):
    i_j = df_matrix[[i, j]].dropna()
    if i_j.empty:
        return 0.0

    a = i_j[i].values
    b = i_j[j].values

    dot_product = np.sum(a * b)
    panjang_a = np.sqrt(np.sum(a ** 2))
    panjang_b = np.sqrt(np.sum(b ** 2))

    if panjang_a == 0 or panjang_b == 0:
        return 0.0

    return dot_product / (panjang_a * panjang_b)


K = 5
hasil_rating = {}

for t in tempat_suka:
    sim_t = pd.Series({j: cosine_similarity(df_matrix, t, j) for j in place if j != t})
    top5 = sim_t.sort_values(ascending=False).head(K)

    rata2_tempat = df_matrix[top5.index].mean()         
    if user_ada:
        rating_user = df_matrix.loc[user, top5.index]    
    else:
        rating_user = pd.Series(np.nan, index=top5.index)

    rating_tetangga = rating_user.fillna(rata2_tempat)
    sumber = np.where(rating_user.notna(), "rating user", "rata-rata tempat")

    valid = rating_tetangga.dropna()
    sim_valid = top5[valid.index]

    if sim_valid.sum() == 0:
        print("Rating tidak bisa dihitung (semua kemiripan 0)")
    else:
        rating = (sim_valid * valid).sum() / sim_valid.sum()
        print({round(rating)})
        hasil_rating[t] = rating


df_matrix_baru = df_matrix.copy()
if not user_ada:
    df_matrix_baru.loc[user] = np.nan     

for t, r in hasil_rating.items():
    df_matrix_baru.loc[user, t] = r

print("\nRating user di pivot table:")
print(df_matrix_baru.loc[[user]].dropna(axis=1))