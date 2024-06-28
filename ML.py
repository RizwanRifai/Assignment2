import pandas as pd #menambahkan library untuk memanipulasi data
import matplotlib.pyplot as plt #menambahkan library untuk memvisualisasikan data
import seaborn as sb #library tambahan untuk matplotlib, berfungsi untuk visualisasi lebih detail
from sklearn.model_selection import train_test_split #membagi dataset menjadi dua subset, yaitu training(latih) set dan test(uji) set
from sklearn.preprocessing import StandardScaler #mengubah data sehingga rata-ratanya nol dan standar deviasi satu
from sklearn.linear_model import LogisticRegression #digunakan untuk masalah klasifikasi
#Menggunakan Logistic Regression karena tujuan utamanya menentukan kategori atau kelas dari sampel input
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report #untuk mengevaluasi kinerja model

# Mengambil file CSV dari penyimpanan internal
path_dasar = r"C:\Users\ASUS\Downloads\ai4i2020.csv"

# Membaca data dari file CSV
isi = pd.read_csv(path_dasar)
print(isi.head())

# Menyiapkan variabel dependen (target) dan independen (fitur)
Y = isi['Machine failure']  # Hanya menggunakan kolom 'Machine failure' sebagai target
X = isi[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']] #Menentukan variabel Independen
Z = isi[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']] #Menentukan variabel dependen

# Menampilkan variabel independen dan dependen
print("\n Variabel Independen (Fitur):")
print(X) #Mencetak variabel independen
print("\n Variabel Dependen (Target):")
print(Z) #Mencetak variabel dependen

# Mengecek apakah ada baris yang duplikat
duplikat = isi[isi.duplicated()]
print("Baris yang duplikat:")
print(duplikat) #Mencetak jumlah baris yang terduplikat

# Menghapus baris yang terduplikat dan memperbarui DataFrame
isi.drop_duplicates(inplace=True)

# Menggabungkan kembali X dan Y untuk menghitung korelasi
data_combined = pd.concat([X, Y], axis=1)
# concat adalah fungsi dalam library pandas yang digunakan untuk menggabungkan dua atau lebih objek pandas
# X merupakan data independen, Y adalah data dependen(Yang dikhususkan untuk Machine failure)
# axis=1 berfungsi untuk menambahkan kolom baru di sebelah kolom yang ada

# Menghitung matriks korelasi
corr_matrix = data_combined.corr()

# Menampilkan matriks korelasi
print("\nKedekatan:")
print(corr_matrix)

# Visualisasi matriks korelasi menggunakan heatmap
plt.figure(figsize=(12, 10))
heatmap = sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrix Hubungan') #Judul dari visual data yang ditampilkan
plt.show() #Mencetak visualisasi data matriks dalam bentuk tabel heatmap

# Membagi data(X dan y) menjadi data latih dan uji
X_latih, X_uji, y_latih, y_uji = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler() #mengubah data sehingga memiliki rata-rata nol dan standar deviasi satu
X_latih_scaled = scaler.fit_transform(X_latih) #fit berfungsi untuk menghitung rata-rata dan standar deviasi dari fitur dalam X_latih
X_uji_scaled = scaler.transform(X_uji)#transform berfungsi untuk mengubah setiap fitur sehingga memiliki rata-rata nol dan standar deviasi satu.

# Deklarasi model Logistic regression
model = LogisticRegression()

# melatih model dengan training set
model.fit(X_latih_scaled, y_latih)

# Memprediksi data uji
y_prediksi = model.predict(X_uji_scaled) #digunakan untuk memprediksi kelas pada test set
X_uji['Prediksi_TWF'] = y_prediksi
print(X_uji) #Mencetak model yang diprediksi

# Mengukur kinerja model
akurasi = accuracy_score(y_uji, y_prediksi) #untuk proporsi prediksi yang benar dari total prediksi
presisi = precision_score(y_uji, y_prediksi)  #proporsi prediksi positif yang benar dari total prediksi positif
pemanggilan = recall_score(y_uji, y_prediksi)    #proporsi prediksi positif yang benar dari total data yang positif(Matriks tertinggi ada pada recall)
conf_matrix = confusion_matrix(y_uji, y_prediksi)  
#mengidentifikasi model kita cenderung membuat prediksi positif palsu (false positive) atau negatif palsu (false negative)
#memberikan gambaran bagaimana model melakukan prediksi terhadap data uji
class_lapor = classification_report(y_uji, y_prediksi)#Menyediakan laporan

# Mencetak kinerja model
print(f"\nAccuracy: {akurasi}") 
print(f"Precision: {presisi}") 
print(f"Recall: {pemanggilan}") 
print("\nConfusion Matrix(Pembanding):") 
print(conf_matrix)
print("\nClassification Report:") 
print(class_lapor)
