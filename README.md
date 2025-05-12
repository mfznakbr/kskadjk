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
- Bagaimana memprediksi nilai penjualan harian pada toko ritel dengan mempertimbangkan fitur yang tersedia?

### Goals
- Membangun model machine learning yang mampu memprediksi penjualan harian toko.

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

### MinMaxScaller 
1. Proses yang dilakukan.
   MinMaxScaler mengubah skala fitur agar berada dalam rentang tertentu, biasanya antara 0 dan 1. Caranya:
   - Hitung nilai minimum dan maksimum dari setiap fitur dalam dataset.
   - Untuk setiap nilai fitur, kurangi dengan nilai minimum fitur tersebut.
   - Kemudian, bagi hasilnya dengan selisih antara nilai maksimum dan minimum fitur tersebut.
2. Alasan :
   Menyamakan Skala: Fitur dengan rentang nilai yang berbeda dapat membingungkan algoritma ML. MinMaxScaler memastikan semua fitur memiliki skala yang sama.
   
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
  
## EVALUATION
Dalam evaluasi model prediksi penjualan harian, beberapa metrik evaluasi digunakan untuk mengukur performa model yang dibangun. Beberapa metrik ini dipilih karena relevansinya dalam konteks prediksi nilai kontinu (yang dalam hal ini, penjualan) dan kemampuan memberikan wawasan berbeda tentang kualitas prediksi.
**Metrik Yang Digunakan** : 
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (Koefisien Determinasi)

**Hasil Evaluasi Proyek :**
Berdasarkan evaluasi pada data train dan test, performa kedua model sebagai berikut : 
Random Forest Regressor:
- Train MSE: 0.00138
- Test MSE: 0.01001
- Train MAE: 0.02694
- Test MAE: 0.07393
- Train R-squared: 0.9606
- Test R-squared: 0.7199

XGBoost Regressor:
- Train MSE: 0.00622
- Test MSE: 0.00782
- Train MAE: 0.05933
- Test MAE: 0.06671
- Train R-squared: 0.8221
- Test R-squared: 0.7812

**Interpretasi Hasil :**
1. Performa Error (MSE dan MAE):
   - Random forest menunjukan performa sangat baik pada train dengan nilai MSE dan MAE sangat rendah. Namun, terdapat peningkatan error signifikan pada data test, terindikasi adanya overfit
   - XGBoost memiliki nilai MSE dan MAE lebih stabil antara data train dan test, berarti model dapat melakukan generalisasi dengan baik.
2. R-squared :
   - Random forest mampu mendapat sebagian besar variabilitas data train mencapai 0.96, tetapi tidak berbeda dengan sebelumnya pada R-squared Random Forest juga terindikasi overfit karena pada data test hanya mencapai 0.71
   - XGBoost mungkin tidak setinggi Random Forest pada train dimana XGBoost hanya mencapai 0.82, namun lebih stabil dimana pada data test 0.78, menunjukan kemampuan baik dalam menangkap pola data.
  
**Kesimpulan**
Berdasarkan metrik evaluasi yang digunakan, XGBoost dapat dianggap sebagai model lebih baik dan stabil dalam konteks ini, karena kemampuannya untuk memberikan prediksi akurat dan konsisten pada data yang belum pernah dilihat sebelumnya.

Metrik yang dipilih juga sesuai dengan problem statement dimana MSE/MAE langsung mengukur seberapa jauh prediksi menyimpang dari nilai aktual penjualan. Sesuai untuk masalah regresi numerik seperti prediksi penjualan. R2 menjelaskan seberapa baik model memperhitungkan variasi data penjualan harian yang dipengaruhi berbagai fitur pada dataset. Penjualan harian adalah variabel kontinu sehingga butuh metrik regresi (bukan klasifikasi seperti accuracy).

Kemudian, metrik ini dipilih karena MSE memberikan interpretasi langsung ("Rata-rata selisih prediksi dengan aktual"), R2 menunjukan apakah fitur yang digunakan benar-benar mempengaruhi prediksi. Serta membandingkan secara objektif performa dua model yang berbeda (RF vs XGBoost).

**Rubik Tambahan**
### Metrik Evaluasi yang Digunakan
Dalam proyek prediksi penjualan harian toko ritel ini, saya menggunakan tiga metrik evaluasi utama:

1. Mean Squared Error (MSE) :
   - Cara Kerja : MSE mengukur rata - rata dari kuadrat selisih antara nilai penjualan aktual dan nilai penjualan yang diprediksi. Prosesnya adalah :
     1. Hitung selisih (error) antara setiap nilai aktual dan nilai prediksi.
     2. Kuadratkan setiap error ini, pengkuadratan memiliki dua tujuan yaitu :
        * Menghilangkan tanda negatif, sehingga error positif dan negatif tidak saling menghilangkan.
        * Memberikan bobot yang lebih besar kepada error yang lebih besar. Ini berarti MSE sangat sensitif terhadap prediksi yang jauh meleset.
      3. Hitung rata - rata dari semua error kuadrat tersebut.
   - Penjelasan : MSE memberikan gambaran tentang variasi rata-rata dari prediksi model Anda. Nilai MSE yang lebih rendah menunjukkan bahwa prediksi model Anda secara rata-rata lebih dekat ke nilai aktual. Namun, karena MSE dalam satuan kuadrat dari variabel target (misalnya, USD^2 jika targetnya adalah penjualan dalam USD), interpretasi langsung dalam satuan target mungkin kurang intuitif.

2. Mean Absolut Error (MAE):
   - Cara Kerja :
     MAE mengukur rata - rata dari nilai absolut selisih antara nilai penjualan aktual dan nilai penjualan prediksi. Prosesnya :
     1. Hitung selisih (error) antara setiap nilai aktual dan nilai prediksinya.
     2. Ambil nilai absolut dari setiap error. ini mengubah semua error jadi positif, sehingga kita hanya mempertimbangkan besarnya error, bukan arahnya.
     3. Hitung rata - rata dari semua nilai absolut error tersebut.
   - Penjelasan : MAE lebih mudah di interpretasikan dibanding MSE karena berada dalam satuan yang sama dengan variabel target (misalnya, Rp) . MAE sebesar Rp.1000 USD berarti bahwa, rata-rata, prediksi model meleset sebesar Rp.1000 dari nilai penjualan aktual. MAE kurang sensitif terhadap outlier dibandingkan MSE, karena tidak mengkuadratkan error.

3. R-squared (R2) :
   - Cara Kerja :
     R-squared mengukur seberapa baik model regresi dalam menjelaskan variabilitas data dependen. Ini membandingkan performa model Saya dengan model *baselline* sederhana yang hanya memprediksi rata - rata nilai target. Prosesnya :
     1. Hitung total variabilitas data target (SST - Total Sum of Squares). ini mengukur seberapa banyak nilai target bervariasi dari rata-ratanya.
     2. Hitung variabilitas yang *tidak* dijelaskan oleh model Anda (SSE - Sum of Squared Errors). Ini mengukur seberapa banyak prediksi model saya meleset dari nilai target.
     3. Hitung proporsi variabilitas yang dijelaskan oleh model. Ini adalah 1 dikurangi rasio SSE terhadap SST.
    
   - Penjelasan :
     * R² mendekati 1: Model menjelaskan sebagian besar variabilitas data. Prediksi model cocok dengan data aktual.
     * R² mendekati 0 : Model tidak dapat menjelaskan banyak variabilitas data. Model mungkin tidak lebih baik dari sekedar memprediksi rata-rata.
     * R² bisa negatif : Model sangat buruk dalam memprediksi data.
