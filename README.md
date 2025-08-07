# 🏡 Ridge & Lasso Regression on Boston Housing Dataset

## 📌 Project Overview

Proyek ini merupakan bagian dari tugas pembelajaran regresi, dengan fokus pada penerapan algoritma **Ridge Regression** dan **Lasso Regression** untuk memprediksi harga rumah berdasarkan dataset **Boston Housing**.

Model yang dikembangkan bertujuan untuk memahami:
- Pengaruh regularisasi dalam regresi linier.
- Perbandingan performa Ridge dan Lasso dalam menghindari overfitting.
- Evaluasi model berdasarkan metrik seperti MAE, MSE, dan R² Score.

## 📂 Dataset

Dataset yang digunakan adalah **Boston Housing Dataset** yang tersedia secara publik melalui `sklearn.datasets`. Dataset ini berisi berbagai fitur properti dan lingkungan, seperti:
- Jumlah kamar
- Jarak ke pusat bisnis
- Pajak properti
- Rasio guru-murid
- Indeks kriminalitas, dll.

## ⚙️ Langkah-Langkah Analisis

1. **Import Library**
2. **Load dan Eksplorasi Dataset**
3. **Pisahkan Fitur dan Target**
4. **Split Data menjadi Train, Validation, dan Test**
5. **Bangun Model Ridge dan Lasso**
6. **Tuning Hyperparameter Alpha**
7. **Evaluasi Model dengan MAE, MSE, dan R²**
8. **Visualisasi Hasil dan Koefisien**

## 📈 Hasil & Insight

- Model **Ridge** memberikan hasil yang lebih stabil dengan fitur yang memiliki multikolinearitas tinggi.
- Model **Lasso** secara otomatis melakukan feature selection dengan mengeliminasi koefisien tidak penting menjadi nol.
- Dengan tuning alpha yang tepat, kedua model dapat menghindari overfitting dan menghasilkan generalisasi yang baik.
- Ridge cocok jika ingin mempertahankan semua fitur, Lasso cocok jika ingin model yang lebih simpel dan interpretatif.

## 💡 Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## 📁 File Terkait

- `Ridge & Lasso Regression on Boston Housing Dataset.ipynb` – notebook utama proyek
- `Ridge & Lasso Regression on Boston Housing Dataset.py` – versi script Python
- `boston.csv` – salinan dataset

---

> Proyek ini dikerjakan oleh **Dimas Adi Prasetyo** sebagai bagian dari pembelajaran Machine Learning.
