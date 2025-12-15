# 🌸 Skincare Recommendation System - Deployment Guide

## 📋 Prerequisites

1. Python 3.8 atau lebih tinggi
2. Semua dependencies yang terinstall

## 🚀 Langkah-langkah Deployment

### 1. Export Model dan Data

Jalankan cell terakhir di notebook `main.ipynb` untuk export semua file yang dibutuhkan:

```python
# Cell akan membuat folder 'deployment_files' dan menyimpan:
# - skincare_model.h5 (model klasifikasi)
# - embedding_model.h5 (model ekstraksi embedding)
# - product_embeddings.npy (embedding produk)
# - similarity_matrix.npy (matriks similarity)
# - skincare_products.csv (data produk)
# - tokenizer.pkl (tokenizer)
# - label_encoder.pkl (label encoder)
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit

```powershell
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur File

```
Tubes DL/
├── main.ipynb                      # Notebook utama
├── app.py                          # Aplikasi Streamlit
├── requirements.txt                # Dependencies
├── README_DEPLOYMENT.md           # Panduan ini
└── deployment_files/              # Folder hasil export
    ├── skincare_model.h5          # Model klasifikasi
    ├── embedding_model.h5         # Model embedding
    ├── product_embeddings.npy     # Embedding produk
    ├── similarity_matrix.npy      # Matriks similarity
    ├── skincare_products.csv      # Data produk
    ├── tokenizer.pkl              # Tokenizer
    └── label_encoder.pkl          # Label encoder
```

## 🎨 Fitur Aplikasi

1. **Pencarian Produk**: Cari produk berdasarkan nama
2. **Filter Brand**: Filter rekomendasi berdasarkan brand tertentu
3. **Filter Tipe Kulit**: Filter berdasarkan jenis kulit
4. **Top N Rekomendasi**: Pilih jumlah rekomendasi yang ditampilkan
5. **Detail Produk**: Menampilkan bahan aktif, manfaat, dan similarity score

## 🎯 Cara Menggunakan Aplikasi

1. **Pilih Produk**: Gunakan dropdown atau search box untuk memilih produk
2. **Atur Filter**: (Optional) Pilih filter brand atau tipe kulit
3. **Tentukan Jumlah**: Pilih berapa banyak rekomendasi yang ingin ditampilkan
4. **Klik "Cari Rekomendasi"**: Sistem akan menampilkan produk yang serupa

## 🌐 Deploy ke Cloud (Optional)

### Streamlit Cloud

1. Push code ke GitHub repository
2. Kunjungi [share.streamlit.io](https://share.streamlit.io)
3. Connect repository Anda
4. Deploy aplikasi

### Heroku

```powershell
# Install Heroku CLI terlebih dahulu
heroku login
heroku create skincare-recommender
git push heroku main
```

## ⚠️ Troubleshooting

### Error: "Model files not found"
- Pastikan sudah menjalankan cell export di notebook
- Periksa folder `deployment_files` sudah berisi semua file

### Error: "Module not found"
- Jalankan: `pip install -r requirements.txt`

### Port sudah digunakan
- Hentikan aplikasi Streamlit yang sedang berjalan
- Atau gunakan port lain: `streamlit run app.py --server.port 8502`

## 📊 Model Performance

Model ini menggunakan:
- **Embedding Dimension**: 128
- **Product Embedding**: 32D
- **Training Epochs**: 30
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)

## 🎨 Design Features

- Tema pink/purple gradient sesuai dunia kecantikan
- Card-based UI untuk tampilan produk
- Responsive design untuk berbagai ukuran layar
- Similarity score dengan warna gradient
- Icon dan emoji untuk visual menarik

## 📞 Support

Jika ada pertanyaan atau masalah, pastikan:
1. Semua file di folder `deployment_files` lengkap
2. Dependencies terinstall dengan benar
3. Python version kompatibel (3.8+)
