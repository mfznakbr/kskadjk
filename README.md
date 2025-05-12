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

