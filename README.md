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
