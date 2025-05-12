# Laporan Proyek Machine Learning - Muhammad Fauzani Akbar

## Project Overview

## Latar Belakang 
Prediksi penjualan merupakan elemen penting dalam pengambilan keputusan yang cukup strategis pada bisnis ritel. Tanpa perkiraan yang akurat, toko berisiko mengalami kelebihan atau kekurangan stok, serta kesulitan dalam perencanaan operasional toko.

Pendekatan machine learning kini menjadi pilihan yang populer karena mampu mengidentifikasi pola yang kompleks dalam data dan memberikan prediksi dengan akurasi tinggi. Mustapha dan Sithole(2025) membuktikan bahwa algoritma seperti XGBoost menghasilkan performa bagus dalam memprediksi penjualan, dengan nilai evaluasi seperti RMSE dan MAE yang lebih baik dibandingkan metode tradisional

Proyek ini bertujuan untuk membangun model prediksi penjualan harian toko ritel berdasarkan fitur - fitur yang ada pada dataset, guna membantu pengambilan keputusan yang lebih efektif.

### Alasan dan Cara Penyelesaian Masalah
Masalah prediksi penjualan perlu diselesaikan karena kesalahan estimasi dapat menyebabkan kerugian, baik dari stok maupun biaya operasional. Dengan menggunakan machine learning, fitur - fitur pada dataset seperti competition distance, promo, holiday, dapat dianalisis secara otomatis untuk menghasilkan prediksi lebih akurat dan efisien. Pendekatan ini terbukti seperti dijelaskan oleh Mustapha dan Sithole (2025), dimana model XGBoost memberikan performa prediksi terbaik.

### Refrensi
O. O. Mustapha and T. Sithole, “Forecasting retail sales using machine learning models,” *Am. J. Stat. Actuar. Sci.*, vol. 6, no. 1, pp. 35–67, Jan. 2025, doi: [10.47672/ajsas.2679](https://doi.org/10.47672/ajsas.2679).

## 🎽 Bussines Understanding

Dalam bisnis ritel, kemampuan untuk memprediksi penjualan secara akurat sangat penting untuk pengambilan keputusan strategis, seperti pengelolaan stok, penjadwalan karyawan, dan perencanaan promosi. Rossmann, sebagai jaringan toko ritel, memiliki data penjualan historis yang kaya dan dilengkapi dengan informasi promosi, tipe toko, serta keberadaan kompetitor yang dapat dimanfaatkan untuk membangun model prediktif.

### Problem Statement
- Bagaimana memprediksi nilai penjualan harian pada toko ritel dengan mempertimbangkan berbagai fitur?

### Goals
- Membangun model machine learning yang mampu memprediksi penjualan harian toko secara akurat.
- Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap penjualan, seperti pengaruh promosi dan kompetitor.

### Solution Approach

#### Solution Statements
- **Random Forest Regressor :** Algoritma ensemble mampu menangani data numerik dan kategorikal, serta memberikan interpretasi melalui feature importance
- **XGBoost Regressor** : Algoritma gradient boosting yang terbukti sangat akurat untuk data tabular dan sering digunakan dalam kompetisi prediksi, termasuk kasus retail sales.

## 📊 Data Understanding
sumber data : (Kaggle [Rossman Store Sales] : https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales)

### Informasi Umum :
- Jumlah baris (setelah sampling dan dropna) : 29.894
- Jumlah kolom : 15
- Tipe data : Kombinasi numerik, kategorikal, dan waktu
- Target Variabel : "Sales" (Penjualan harian per toko)

### Daftar Fitur
- `Store` : ID unik toko
- `DayOfWeek` : Hari dalam minggu (1 = Senin, 7 = Minggu)
- `Sales` : Total penjualan pada hari tertentu (target prediksi)
- `Promo` : Apakah toko sedang melakukan promosi (1 = Ya, 0 = Tidak)
- `SchoolHoliday` : Apakah hari tersebut bertepatan dengan libur sekolah
- `StoreType` : Tipe toko (a, b, c, d)
- `Assortment` : Tingkat kelengkapan produk (a = basic, b = extra, c = extended)
- `CompetitionDistance` : Jarak (meter) ke toko pesaing terdekat
- `Promo2` : Apakah toko ikut serta dalam promosi jangka panjang Promo2
- `Day`, `Month` : Informasi tanggal dari kolom `Date`
- `CompetitionOpenSince` : Lama kompetitor telah buka (dalam bulan)
- `Promo2InMonth` : Apakah bulan saat ini termasuk dalam jadwal promosi (1/0)
- `PromoMonth` : Apakah saat ini promosi Promo2 sedang aktif (1/0)

**Rubik Tambahan**
### Menangani missing value
- Data missing value menggunakan isna().sum() mencapai 508.031 pada Promo2SinceWeek, dan 2 lainnya.
- Namun untuk mengatasi missing value tidak dapat langsung menggunakan dropna() karena akan menghilangkan nilai b pada fitur assorment dan storetype.
- mengindetinfikasi nilai 0 pada kolom sales, jumlah nilai 0 dari sales mencapai 172.871 dan setelah men-Drop nilai 0 pada sales nilai min pada sales itu 46 dan max adalah 41551.

### Menangani Outliers
- Disini, saya hanya akan mengatasi outliers pada kolom sales dan competition distance. Karena, kolom lain itu hanya bernilai 0-1, tahun, jadi tidak perlu dilakukan penanganan outlier
- Untuk mengatasi outlier, saya menggunakan metode IQR dan setelah mengatasi outlier menggunakan IQR jumlah data tersisa 732.741

### Data Cleaning
- Membuat kolom baru yaitu CompetitionOpenSince dari hasil kalkulasi CompetitionOpenSinceYear dan CompetitionOpenSinceMonth. Ini berfungsi untuk mengetahui berapa lama saingan membuka tokonya dalam hitungan bulan.
- Membuat kolom baru Promo2InMonth dari hasil kalkulasi Promo2SicneYear dan Promo2SinceWeek.
- Membuat kolom baru PromoMonth hasil maping MonthName dan kolom PromoInterval.
- Menghapus kolom Open : dilihat dari hasil describe telihat bahwa tidak ada nilai 0 yang menandakan tutup dan juga saya telah menghapus nilai 0 pada sales.

### Univariate Analysis
- melakukan pembagian fitur pada dataset numeric dan categoric
- Distribusi Stateholiday : didominasi oleh 0 dimana artinya itu bukan hari libur sedangkan hari libur sendiri sangat kecil perbandingannya dengan hari biasa.
- Distribusi StoreType : jumlah data tertinggi ada pada tipe toko a yaitu 15.878 dan minimal tipe toko b yaitu 479.
- Distribusi Assorment : jumlah data tertinggi adalah a yaitu 16.278 dan terendah yaitu b 287.
- Distribusi Numeric : disini saya hanya akan fokus pada distribusi fitur sales, dimana terlihat pada awalnya cenderung naik hingga pada titik tertinggi sekitar 1300+ dan turun terus hingga dibawah 200.

### Multivariate Analysis
1. Categorical Feature
   - StoreType 'b' memiliki rata-rata penjualan (Sales) yang lebih tinggi dibandingkan StoreType 'd' (terlihat dari posisi batang/garis yang lebih tinggi untuk 'b').
   - Jika perbedaan tinggi batang/garis antara 'd' dan 'b' besar, berarti tipe toko ('StoreType') berpengaruh kuat terhadap penjualan.

## 🎰 Data Preparation
- Saya melakukan teknik yang umum digunakan untuk encoding fitur kategori yaitu teknik one-hot-encoding. Men-encode tiga variabel yaitu "StoreTyoe", "Assortment", "StateHoliday"
- Saya tidak melakukan reduksi PCA karena komponen perta (PC1) hampir mencakup seluruh informasi penting dalam data. Dan PC2 hanya menambahkan 0.73% informasi, yang bisa diabaikan.
- Melakukan spliting test dan train dengan pembagian data 80 : 20.
- Melakukan standarisasi dengan MinMaxScaler

**Rubik Tambahan**
### Encoding
1. Proses yang dilakukan (one-hot-encoding) :
   - One-Hot Encoding: Ubah kolom kategorikal (StateHoliday, StoreType, Assortment) jadi kolom biner (0/1).
   - Gabung Kolom Baru: Tambahkan kolom hasil encoding ke DataFrame.
   - Hapus Kolom Asli: Kolom kategorikal lama dihapus karena sudah diencode.
2. Alasan :
   - Model machine learning butuh data numerik, bukan kategori (seperti a, b).
   - Hindari bias ordinal (misal: anggap a=1, b=2 lebih penting padahal tidak).
   - Biarkan model belajar pola tiap kategori secara terpisah.

### Teknik Spliting
1. Proses yang dilakukan :
   - X = sampel.drop('Sales', axis=1) → Semua kolom kecuali Sales sebagai fitur. y = sampel['Sales'] > Kolom Sales sebagai target.
   - Test 20%: Data uji = 20% dari total, data latih = 80%. random_state=123: Memastikan pembagian sama tiap kali di-run (reproducibility).
2. Alasan :
   - Mengevaluasi performa model secara objektif pada data yang belum pernah dilihat (data test).
   - Mencegah overfitting dengan memisahkan data validasi.

### Convert Float ke Int
1. Proses yang dilakukan :
   - Mengumpulkan kolom yang memiliki tipe data float dalam variabel kolom_yang_diubah
   - lalu mengubahnya ke tipe data int dengan fungsi astype() dan disimpan ke dalam dataset sampel.
2. Alasan  :
   - Untuk mengubah float ke int agar semua data bertipe int.
  
## MODELING
**TOP Rekomendasi Penjualan Tertinggi Menggunakan XGBoost**
1. Top 1 terbaik adalah penjualan dengan total 1.147878 pada store 567 
2. Total Penjualan terendah adalah 0.108608 pada store  510   
   
**Terdapat pola kunci naik turunya penjualan**
- Promo = Penjualan naik
- Hari Kerja = Penjualan juga naik
- Kompetitor cukup dekat = lebih menguntungkan

**Rekomendasi Bisnis**
- Prioritaskan promo di toko dengan performa buruk
- Fokus stok tambahan ketika menerapkan promo

### Tambahan 
**Saya menggunakan dua algoritma berbeda yaitu Random Forest dan XGBoost**

**Kekurangan dan Kelebihan Random Forest**
1. Kekurangan
   - kurang intuitif untuk rekomendasi
   - Lebih lambat untuk dataset besar
2. Kelebihan
   - Lebih stabil (MSE/RMSE/MAE lebih rendah)
   - Interpretasi lebih mudah
  
**Kekurangan dan Kelebihan XGBoost**
1. Kelebihan
   - R2 tinggi (78.1% di test) -> cocok untuk rekomendasi
   - cepat dan efisien
   - handle missing value otomatis
2. Kekurangan
   - rentan overfitting
   - butuh tuning hyperparameter
