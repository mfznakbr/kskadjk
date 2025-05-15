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
- Jumlah baris (untuk store_df dan train_df) : store_df = 1115 dan train df = 1.017.209
- Jumlah kolom : store_df = 10 dan train_df = 9
- Tipe data : Kombinasi numerik, kategorikal, dan waktu
pada store_df  memiliki 1115 baris dan 10 kolom. Beberapa kolom seperti CompetitionOpenSinceMonth, Promo2SinceWeek, dan PromoInterval memiliki data yang hilang (missing values) yang perlu ditangani pada tahap preprocessing, untuk train_df memiliki 1.017.209 baris dan 9 kolom. Tidak ada missing value. Kolom seperti Sales dan Customers merupakan target dan indikator penting dalam analisis penjualan.

### Daftar Fitur
**TRAIN_DF (train_df)**
| Nama Variabel               | Deskripsi                                                                         |
| --------------------------- | --------------------------------------------------------------------------------- |
| `Store`                     | ID unik toko. Digunakan untuk join ke `train.csv`.                                |
| `StoreType`                 | Tipe toko: `a`, `b`, `c`, `d` (mewakili model bisnis yang berbeda).               |
| `Assortment`                | Tingkat kelengkapan produk:<br>• `a` = basic<br>• `b` = extra<br>• `c` = extended |
| `CompetitionDistance`       | Jarak (dalam meter) ke toko pesaing terdekat.                                     |
| `CompetitionOpenSinceMonth` | Bulan saat kompetitor mulai buka.                                                 |
| `CompetitionOpenSinceYear`  | Tahun saat kompetitor mulai buka.                                                 |
| `Promo2`                    | Apakah toko ikut promosi jangka panjang Promo2 (1 = ya, 0 = tidak).               |
| `Promo2SinceWeek`           | Minggu kalender saat toko mulai ikut Promo2.                                      |
| `Promo2SinceYear`           | Tahun saat toko mulai ikut Promo2.                                                |
| `PromoInterval`             | Bulan-bulan saat Promo2 aktif, contoh: `"Feb,May,Aug,Nov"`.                       |


**STORE_DF (store_df)**
| Nama Variabel               | Deskripsi                                                                         |
| --------------------------- | --------------------------------------------------------------------------------- |
| `Store`                     | ID unik toko. Digunakan untuk join ke `train.csv`.                                |
| `StoreType`                 | Tipe toko: `a`, `b`, `c`, `d` (mewakili model bisnis yang berbeda).               |
| `Assortment`                | Tingkat kelengkapan produk:<br>• `a` = basic<br>• `b` = extra<br>• `c` = extended |
| `CompetitionDistance`       | Jarak (dalam meter) ke toko pesaing terdekat.                                     |
| `CompetitionOpenSinceMonth` | Bulan saat kompetitor mulai buka.                                                 |
| `CompetitionOpenSinceYear`  | Tahun saat kompetitor mulai buka.                                                 |
| `Promo2`                    | Apakah toko ikut promosi jangka panjang Promo2 (1 = ya, 0 = tidak).               |
| `Promo2SinceWeek`           | Minggu kalender saat toko mulai ikut Promo2.                                      |
| `Promo2SinceYear`           | Tahun saat toko mulai ikut Promo2.                                                |
| `PromoInterval`             | Bulan-bulan saat Promo2 aktif, contoh: `"Feb,May,Aug,Nov"`.                       |

**Rubik Tambahan**
🔍 Exploratory Data Analysis (EDA)
Untuk memahami karakteristik toko secara keseluruhan, dilakukan analisis eksploratif terhadap dua fitur kategorikal penting: StoreType dan Assortment.

🏪 StoreType : 
| StoreType | Jumlah Toko | Persentase |
| --------- | ----------- | ---------- |
| a         | 602         | 53.99%     |
| d         | 348         | 31.21%     |
| c         | 148         | 13.27%     |
| b         | 17          | 1.52%      |

**Insight :**
- Mayoritas toko termasuk dalam tipe a. menunjukan tipe ini adalah model toko paling umum digunakan.
- Tipe b sangat jarang digunakan, ditemukan hanya sekitar 1.5% dari total toko.

🛒 Assortment
| Assortment | Jumlah Toko | Persentase |
| ---------- | ----------- | ---------- |
| a          | 593         | 53.18%     |
| c          | 513         | 46.01%     |
| b          | 9           | 0.81%      |

**Insight :**
- Tipe a (basic assortment) merupakan yang paling umum digunakan, diikuti oleh c (extended).
- Tipe b (extra assortment) sangat jarang digunakan, hanya di bawah 1% toko, menunjukkan bahwa hanya sedikit toko yang menawarkan jenis produk lebih banyak dari biasanya.


## 🎰 Data Preparation
- Merge Data
- Penanganan Missing Values
  * Hapus nilai 0 pada kolom sales
  * Penanganan Outliers
  * Data Cleaning (Kalkulasi kolom untuk menghasilkan kolom baru)
- Encoding kategori fitur
- MinMaxScaler 
- Teknik Spliting (Train - test split)
- Convert Float ke Int
  

**Rubik Tambahan**
### Merge Data
1. Proses
   - pd.merge() digunakan untuk menggabungkan train_df (data penjualan) dengan store_df (informasi toko) berdasarkan kolom Store.
   - data.info() menampilkan struktur dataset hasil gabungan, termasuk tipe data dan jumlah nilai yang tidak null.
2. Alasan :
   - Agar informasi dalam dataframe lengkap dan prediksi dapat dilakukan dengan lebih akurat.
     
### Missing Value 
1. Proses 
   Metode penanganan missing values bersamaan dengan :
   - Penanganan Outliers
   - Hapus nilai 0 pada kolom sales
   - Data Cleaning
   - fungsi dropna() setelah **pengambilan sampel data.** 
2. Alasan : Menangani missing value dengan dropna( ) secara langsung sepertinya tidak tepat untuk dataset ini, jika dipaksakan hasilnya akan seperti ini Assorment : ['a' 'c' nan], StoreType : ['a' 'd' 'c' nan] dimana kehilangan nilai b baik pada Assortment dan StoreType. Oleh karena itu, dropna() diterapkan secara selektif dan setelah proses sampling, guna meminimalkan kehilangan variasi kategori yang penting dalam analisis maupun pemodelan.

#### Hapus Nilai 0 Pada Kolom Sales 
1. Proses :
   - Baris data yang memiliki nilai penjualan (Sales) sama dengan nol dihapus dari dataset. Ini dilakukan menggunakan teknik filtering untuk memastikan hanya data dengan transaksi penjualan yang valid yang disertakan.
2. Alasan :
   - Nilai Sales = 0 biasanya menandakan bahwa toko sedang tutup atau tidak melakukan penjualan pada hari tersebut. Jika data ini tetap digunakan, model dapat belajar dari kondisi yang tidak mencerminkan aktivitas penjualan yang sebenarnya. Menghapusnya membantu meningkatkan akurasi dan relevansi model.

#### Penanganan Outliers 
1. Proses :
   - Outlier pada kolom numerikal seperti Sales dan CompetitionDistance dideteksi menggunakan metode Interquartile Range (IQR). Nilai-nilai yang berada jauh di bawah atau di atas sebaran umum dihapus dari dataset.
2. Alasan :
   - Outlier dapat mempengaruhi distribusi data dan menyebabkan model menjadi bias atau overfitting. Dengan menghapus nilai-nilai ekstrem ini, kualitas data menjadi lebih stabil dan representatif terhadap pola penjualan yang umum terjadi.

#### Data Cleaning
1. Proses :
   - kolom Date dipecah menjadi fitur numerik baru: Day, Month, dan Year menggunakan .dt
   - fillna(0) digunakan untuk mengisi nilai missing (NaN) pada kolom CompetitionOpenSinceYear dan CompetitionOpenSinceMonth dengan nilai 0.
   - Kalkulasi dilakukan dengan menghitung selisih waktu antara tahun dan bulan kompetitor mulai buka (CompetitionOpenSinceYear dan CompetitionOpenSinceMonth) dengan waktu transaksi (Year dan Month) pada data.
   - Dibuat fitur baru bernama Promo2InMonth yang menghitung lama waktu (dalam bulan) sejak toko mulai berpartisipasi dalam promosi jangka panjang Promo2.
   - Map angka bulan ke nama bulan (1 → 'Jan', dst).Bandingkan nama bulan transaksi (MonthName) dengan daftar bulan pada kolom PromoInterval.
   - Diambil 30.000 baris data secara acak menggunakan .sample() untuk mengurangi beban komputasi saat eksplorasi atau eksperimen awal.
   - Menghapus sisa missing value pada data sampel menggunakan fungsi dropna()
2. Alasan:
   - Memecah Date menjadi fitur waktu terpisah seperti hari, bulan, dan tahun membantu model dalam menangkap pola musiman atau tren waktu dalam data penjualan.
   - Karena banyak nilai yang hilang pada dua kolom ini, strategi yang dipilih adalah mengganti nilai kosong dengan 0 yang berarti “tidak diketahui/tidak tersedia” daripada membuang data berharga lainnya.
   - Juga memberikan gambaran tahun-tahun pembukaan kompetitor yang tercatat, yang bisa digunakan untuk membuat fitur baru seperti “lama persaingan”.
   - Representasi dalam satuan bulan memungkinkan model menangkap hubungan temporal secara lebih halus dibandingkan dua kolom terpisah.
   - Lama waktu promo berlangsung kemungkinan memengaruhi tingkat penjualan — semakin lama suatu toko berpartisipasi dalam promo, semakin besar kemungkinan pengaruhnya terhadap performa penjualan.
   - Dengan mengekstrak informasi apakah bulan saat ini termasuk bulan promo (PromoMonth), model dapat lebih mudah memahami pengaruh musiman promosi terhadap penjualan.
   - Dataset awal sangat besar (~730.000+ baris), yang bisa membuat proses visualisasi, eksplorasi, atau training model menjadi lambat dan berat.
   - Agar data siap digunakan untuk encoding dan tahap selanjutnya

### Encoding
1. Proses yang dilakukan (one-hot-encoding) :
   - One-Hot Encoding: Ubah kolom kategorikal (StateHoliday, StoreType, Assortment) jadi kolom biner (0/1).
   - Gabung Kolom Baru: Tambahkan kolom hasil encoding ke DataFrame.
   - Hapus Kolom Asli: Kolom kategorikal lama dihapus karena sudah diencode.
2. Alasan :
   - Model machine learning butuh data numerik, bukan kategori (seperti a, b).
   - Hindari bias ordinal (misal: anggap a=1, b=2 lebih penting padahal tidak).
   - Biarkan model belajar pola tiap kategori secara terpisah.]
     
### MinMaxScaller 
1. Proses yang dilakukan.
   MinMaxScaler mengubah skala fitur agar berada dalam rentang tertentu, biasanya antara 0 dan 1. Caranya:
   - Hitung nilai minimum dan maksimum dari setiap fitur dalam dataset.
   - Untuk setiap nilai fitur, kurangi dengan nilai minimum fitur tersebut.
   - Kemudian, bagi hasilnya dengan selisih antara nilai maksimum dan minimum fitur tersebut.
2. Alasan :
   Menyamakan Skala: Fitur dengan rentang nilai yang berbeda dapat membingungkan algoritma ML. MinMaxScaler memastikan semua fitur memiliki skala yang sama.

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
   
## ♣️ MODELING

### 1. **Random Forest Regressor**

**Deskripsi**:
Random Forest adalah algoritma ensemble learning berbasis decision tree. Ia bekerja dengan membangun banyak pohon keputusan (decision trees) secara paralel, lalu menggabungkan hasilnya untuk menghasilkan prediksi yang lebih stabil dan akurat. Untuk regresi, prediksi akhir diambil dari rata-rata hasil semua pohon.

**Parameter yang Digunakan**:

* `n_estimators=100`: Jumlah pohon (trees) yang dibangun dalam model.
* `random_state=123`: Nilai acak untuk memastikan hasil yang konsisten setiap kali model dijalankan.
* `n_jobs=-1`: Menggunakan semua core CPU yang tersedia untuk mempercepat proses training.

**Cara kerja**:

* Setiap pohon dilatih menggunakan subset acak dari data (bootstrap sampling).
* Fitur yang digunakan pada tiap node juga dipilih secara acak.
* Hasil akhir prediksi diambil dari **rata-rata prediksi** seluruh pohon.

---

### 2. **XGBoost Regressor**

**Deskripsi**:
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang membangun pohon keputusan secara bertahap, di mana setiap pohon baru dibentuk untuk memperbaiki kesalahan dari pohon sebelumnya. Cocok digunakan untuk data besar dengan kompleksitas tinggi.

**Parameter yang Digunakan**:

* `n_estimators=300`: Jumlah pohon yang dibangun secara bertahap.
* `max_depth=4`: Maksimal kedalaman setiap pohon. Membatasi kedalaman pohon mencegah overfitting.
* `random_state=42`: Nilai acak untuk reprodusibilitas hasil.
* `n_jobs=-1`: Menggunakan semua core CPU yang tersedia untuk proses training.

**Cara kerja**:

* XGBoost menggunakan metode boosting, yang memperbaiki prediksi dengan menambahkan pohon secara bertahap.
* Setiap pohon baru berusaha meminimalkan error dari pohon sebelumnya menggunakan metode **gradient descent**.
* Model ini memiliki keunggulan dalam performa dan kecepatan training.



**TOP Rekomendasi Penjualan Tertinggi Menggunakan XGBoost**
1. Top 1 terbaik adalah dengan prediksi penjualan  1.526806 pada store 247
2. Top 1 Prediksi Penjualan terendah 0.223762 adalah pada store 310 
   
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
