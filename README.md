# Submission 1: Implementasi Machine Learning Pipeline dengan TFX
Nama: Bernadetta Sri Endah Dwi
Username dicoding: bernadettadwi14@gmail.com

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [iris dataset](https://archive.ics.uci.edu/dataset/53/iris) |
| Masalah | Klasifikasi spesies bunga Iris berdasarkan pengukuran dimensi kelopak dan mahkota bunga. Dataset ini memiliki 3 kelas spesies bunga Iris yang perlu diklasifikasikan: Setosa, Versicolor, dan Virginica. |
| Solusi machine learning | Mengimplementasikan machine learning pipeline menggunakan TensorFlow Extended (TFX) untuk mengotomatisasi proses pengembangan model klasifikasi. Pipeline ini mencakup semua komponen yang diperlukan mulai dari pengimporan data hingga evaluasi dan deployment model. |
| Metode pengolahan | Metode pengolahan data meliputi: normalisasi fitur numerik menggunakan scale_to_0_1 untuk membawa nilai fitur ke rentang [0,1], konversi label kategori ke format numerik (0, 1, 2), dan validasi kualitas data menggunakan ExampleValidator. |
| Arsitektur model | Model menggunakan arsitektur neural network sederhana dengan layer Dense dari TensorFlow Keras, terdiri dari: input layer (4 neuron untuk 4 fitur), 2 hidden layer (masing-masing 10 neuron dengan aktivasi ReLU), dan output layer (3 neuron dengan aktivasi softmax untuk 3 kelas). |
| Metrik evaluasi | Model dievaluasi dengan metrik SparseCategoricalAccuracy, dengan ambang batas (threshold) minimal 0.6 (60%). Evaluasi dilakukan melalui komponen Evaluator yang membandingkan model baru dengan model yang sebelumnya (jika ada). |
| Performa model | Pipeline berhasil dibuat dengan 9 komponen TFX yang berjalan dengan baik. Model dilatih untuk melakukan klasifikasi bunga Iris dengan akurasi di atas ambang batas 60%. Hasil dari pipeline tersimpan di direktori 'bernadetta-pipeline' dengan model yang siap untuk deployment tersimpan di subfolder 'pushed_model'. |
