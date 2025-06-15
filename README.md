# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis
- Jumlah siswa yang mengalami dropout cukup tinggi setiap tahunnya.
- Belum ada sistem prediksi yang dapat mengidentifikasi siswa berisiko dropout lebih awal.
- Kurangnya visibilitas terhadap performa siswa secara menyeluruh dan realtime.

### Cakupan Proyek
- Menganalisis data performa akademik dan latar belakang siswa.
- Membangun model machine learning untuk memprediksi potensi dropout.
- Membuat dashboard interaktif untuk memantau performa siswa dan hasil prediksi.
- Menyediakan prototype aplikasi berbasis web (Streamlit) untuk penggunaan internal oleh staf akademik.

### Persiapan

Sumber data: [Jaya Jaya Institut](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

Setup environment:

1. Unduh _file_ zip dari repositori ini  kemudian ekstrak pada sebuah direktori
2. Buka dan jalankan **Anaconda Prompt**
3. pindah ke direktori tempat menyimpan _files_ yang telah diekstrak tadi
 
```
cd path/to/destination/directory
```
 
4. Membuat environment baru dengan nama sesuai keinginan
 
```
conda create --name <nama-venv> python=3.10
```
 
5. Mengaktifkan environment yang telah dibuat
 
```
conda activate <nama-venv>
```
 
6. _Install_ semua library yang dibutuhkan
 
```
pip install -r requirements.txt
```
 
7. Jalankan jupyter notebook
 
```
jupyter-notebook .
```
 
8. Unggah dataset yang  telah diunduh dan letakkan dalam satu folder dengan berkas _notebook.ipynb_
9. Buka dan jalankan berkas _notebook.ipynb_.

## Business Dashboard
Business dashboard dibangun menggunakan Streamlit yang menampilkan input form, informasi siswa, serta hasil prediksi dropout secara individual. Dashboard ini membantu staf akademik dalam mengidentifikasi siswa yang membutuhkan perhatian khusus.

> Link akses dashboard (jika online): https://public.tableau.com/authoring/iqbaledutechh/Dashboard1#1

## Menjalankan Sistem Machine Learning
Prototype aplikasi machine learning telah dibuat dalam bentuk web app menggunakan Streamlit.

Langkah menjalankan sistem:
1. Buka terminal pada _virtual environment_ yang telah dibuat sebelumnya.
2. Pastikan direktori saat ini menampung berkas-berkas yang telah diekstrak sebelumnya, terutama yang memiliki berkas **app.py**. Jika belum di direktori yang tepat, bisa menggunakan perintah di bawah

```
cd path/to/destination/directory
```

3. Setelah direktorinya sesuai, bisa menjalankan perintah di bawah

```
streamlit run app.py
```

4. Setelah berhasil dijalankan, masukkan data yang sesuai kemudian klik tombol **Predict** untuk mengetahui status siswa tersebut.

Sedangkan untuk menjalankannya melalui link, bisa diakses dengan klik link berikut:
[Akan ditambahkan jika dideploy ke Streamlit Cloud]
## Conclusion
Proyek ini berhasil membangun sistem prediksi dropout yang dapat mengidentifikasi siswa berisiko tinggi berdasarkan data historis akademik dan administratif. Dengan sistem ini, Jaya Jaya Institut dapat mengambil langkah proaktif dalam memberikan intervensi atau bimbingan kepada siswa yang membutuhkan.

### Rekomendasi Action Items
Untuk mengurangi angka siswa yang _dropout_, dapat dilakukan rekomendasi berikut:
- Menerapkan sistem prediksi ini secara internal untuk monitoring siswa secara berkala.
- Memberikan perhatian khusus dan bimbingan tambahan kepada siswa yang diprediksi akan dropout.
- Melakukan evaluasi dan retraining model secara berkala dengan data terbaru agar performa tetap optimal.
- Mengembangkan dashboard batch monitoring untuk melihat status seluruh siswa dalam satu tampilan.

