import math  # digunakan untuk fungsi matematika seperti cos, sin, dll
import numpy as np  # (opsional) untuk operasi numerik, meskipun belum banyak dipakai
import matplotlib.pyplot as plt  # untuk membuat grafik (plot visualisasi)


class NewtonRaphson:
    """
    Class untuk menyelesaikan persamaan nonlinier menggunakan
    metode Newton-Raphson.
    """

    jumlah_objek = 0  # menghitung berapa banyak objek class yang dibuat

    def __init__(self, fungsi, turunan, x0, toleransi=1e-6, maks_iter=50):
        """
        Parameter sesuai soal:
        ----------
        fungsi      : fungsi f(x)
        turunan     : turunan f'(x)
        x0          : tebakan awal
        toleransi   : batas galat
        maks_iter   : jumlah iterasi maksimum
        """
        self.fungsi = fungsi  # menyimpan fungsi f(x)
        self.turunan = turunan  # menyimpan turunan f'(x)
        self.x0 = x0  # tebakan awal
        self.toleransi = toleransi  # batas error (kriteria berhenti)
        self.maks_iter = maks_iter  # maksimum iterasi

        self.iterasi = []  # menyimpan riwayat iterasi (sesuai soal)
        self.akar = None  # menyimpan hasil akar
        self.konvergen = False  # status apakah konvergen atau tidak

        NewtonRaphson.jumlah_objek += 1  # menambah jumlah objek

    def hitung(self):
        """
        Menjalankan iterasi Newton-Raphson.
        (Sesuai soal: method untuk menjalankan iterasi)
        """
        x = self.x0  # mulai dari tebakan awal

        for i in range(1, self.maks_iter + 1):
            fx = self.fungsi(x)  # menghitung f(x)
            dfx = self.turunan(x)  # menghitung f'(x)

            # Encapsulation & Abstraction: Menangani pembagian nol secara internal
            if abs(dfx) < 1e-12: 
                raise ZeroDivisionError(f"Turunan terlalu kecil (mendekati nol) pada iterasi {i}.")

            # rumus utama Newton-Raphson
            x_baru = x - fx / dfx

            # Deteksi Divergensi (Abstraksi)
            if abs(x_baru) > 1e10: # Jika nilai x menjadi terlalu besar
                print("Peringatan: Metode kemungkinan divergen (menjauhi akar).")
                break

            # menghitung galat (error)
            galat = abs(x_baru - x)

            # menyimpan data iterasi (sesuai soal: menyimpan riwayat iterasi)
            self.iterasi.append({
                # append() → menambahkan data baru ke dalam list self.iterasi

                "iterasi": i,
                # menyimpan nomor iterasi ke-i

                "x_lama": x,
                # menyimpan nilai x sebelum diperbarui (nilai lama)

                "f_x": fx,
                # menyimpan nilai fungsi f(x) pada x lama

                "f_aksen_x": dfx,
                # menyimpan nilai turunan f'(x) pada x lama

                "x_baru": x_baru,
                 # menyimpan nilai x hasil perhitungan Newton-Raphson

                "galat": galat
                # menyimpan nilai error (selisih antara x_baru dan x_lama)
            })
            # seluruh data disimpan dalam bentuk dictionary
            # lalu dimasukkan ke dalam list self.iterasi

            # kondisi berhenti jika galat sudah kecil
            if galat < self.toleransi:
                self.akar = x_baru  # simpan akar
                self.konvergen = True  # tandai konvergen
                return x_baru  # mengembalikan akar (sesuai soal)

            x = x_baru  # lanjut ke iterasi berikutnya

        # jika tidak konvergen sampai batas iterasi
        self.akar = x   # menyimpan nilai terakhir sebagai akar pendekatan
        return x    # mengembalikan nilai tersebut sebagai hasil fungsi

    def tampilkan_hasil(self):
        """
        Menampilkan tabel iterasi.
        (Sesuai output: tabel iterasi)
        """
        # pengecekan apakah data iterasi masih kosong
        if not self.iterasi:
            # jika list iterasi kosong (hitung() belum dijalankan)
            print("Belum ada iterasi. Jalankan hitung() terlebih dahulu.")
            return  # menghentikan fungsi agar tidak lanjut ke bawah

        # jika data iterasi ada, maka lanjut menampilkan tabel
        print("=" * 95) # membuat garis pembatas
        print(
            "{:<6}{:<18}{:<18}{:<18}{:<18}{:<15}".format(
                "Iter",     # kolom nomor iterasi
                "x lama",   # nilai x sebelumnya
                "f(x)",     # nilai fungsi f(x)
                "f'(x)",    # nilai turunan f'(x)
                "x baru",   # nilai x hasil iterasi
                "Galat"     # nilai error (selisih)
            )
        )
        print("=" * 95) # garis pembatas bawah header

        # menampilkan setiap iterasi dalam bentuk tabel
        for data in self.iterasi:
            # perulangan (loop) untuk mengambil setiap data iterasi
            # 'data' berisi dictionary dari setiap langkah iterasi
            print(
                # menampilkan data dalam format rapi per kolom
                f"{data['iterasi']:<6}"             # nomor iterasi, rata kiri lebar 6 karakter
                f"{data['x_lama']:<18.10f}"         # nilai x lama, 10 angka desimal, lebar 18
                f"{data['f_x']:<18.10f}"            # nilai f(x), 10 angka desimal
                f"{data['f_aksen_x']:<18.10f}"      # nilai turunan f'(x), 10 angka desimal
                f"{data['x_baru']:<18.10f}"         # nilai x baru hasil iterasi
                f"{data['galat']:<15.10f}"          # nilai galat (error), 10 angka desimal
            )

        # menampilkan garis penutup tabel
        print("=" * 95) # membuat garis horizontal sepanjang 95 karakter

    def analisis(self):
        """
        Menampilkan analisis hasil iterasi.
        (Sesuai output: analisis konvergensi)
        """
        print("\n=== Analisis Konvergensi ===")
        # menampilkan judul analisis ( \n untuk pindah baris)

        if self.konvergen:
            # kondisi jika metode berhasil konvergen
            print("Metode Newton-Raphson berhasil konvergen.")
            # informasi bahwa metode sukses menemukan akar
            print(f"Tebakan awal      : {self.x0}")
            # menampilkan nilai tebakan awal (x0)
            print(f"Akar hampiran     : {self.akar:.10f}")
            # menampilkan hasil akar dengan 10 angka desimal
            print(f"Jumlah iterasi    : {len(self.iterasi)}")
            # menampilkan jumlah iterasi yang dilakukan
            print(f"Toleransi         : {self.toleransi}")
            # menampilkan nilai toleransi yang digunakan
            print(
                f"Galat terakhir    : "
                f"{self.iterasi[-1]['galat']:.10e}"
            )
            # menampilkan galat terakhir
            # [-1] → mengambil data iterasi terakhir
            # .10e → format notasi ilmiah (eksponen)
        else:
            # kondisi jika metode tidak konvergen
            print("Metode tidak konvergen.")
            # informasi bahwa metode gagal
            print(f"Tebakan awal      : {self.x0}")
            # tetap menampilkan tebakan awal
            print(f"Jumlah iterasi    : {self.maks_iter}")
            # menampilkan jumlah iterasi maksimum yang telah dicapai

    def plot_visualisasi(self, rentang=2):
        """
        Menampilkan grafik konvergensi dan grafik fungsi secara vertikal (Atas-Bawah).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        if not self.iterasi:
            print("Jalankan hitung() terlebih dahulu!")
            return

      
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(f'Analisis Metode Newton-Raphson (Kelompok 3)', fontsize=16)

        # --- Grafik 1: Konvergensi (Atas) 
        iterasi_ke = [data["iterasi"] for data in self.iterasi]
        galat = [data["galat"] for data in self.iterasi]
        
        ax1.plot(iterasi_ke, galat, marker='o', color='tab:orange', linewidth=2)
        ax1.set_yscale('log')
        ax1.set_title("Konvergensi Galat (Skala Log)")
        ax1.set_xlabel("Iterasi")
        ax1.set_ylabel("Galat")
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # --- Grafik 2: Visualisasi Fungsi & Akar (Bawah)
        x_min, x_max = self.akar - rentang, self.akar + rentang
        x_plot = np.linspace(x_min, x_max, 500)
        y_plot = [self.fungsi(x) for x in x_plot]

        ax2.plot(x_plot, y_plot, label='f(x)', color='tab:blue', linewidth=2)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.scatter([self.akar], [0], color='red', s=80, zorder=5, label=f'Akar: {self.akar:.4f}')
        
        ax2.set_title("Posisi Akar pada Grafik Fungsi")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Mengatur jarak antar grafik agar label tidak tumpang tindih
        plt.tight_layout(pad=4.0)
        plt.show()

    def get_akar(self):
        """
        Mengembalikan nilai akar yang ditemukan.
        (Sesuai soal: mengembalikan akar pendekatan)
        """
        return self.akar
        # mengembalikan nilai akar yang telah dihitung

    @staticmethod
    def validasi_fungsi(fungsi, x):
        """
        Memeriksa apakah fungsi dapat dihitung pada titik x.
        (Tambahan fitur validasi)
        """
        try:
            fungsi(x)
            # mencoba menjalankan fungsi pada nilai x

            return True
            # jika berhasil, berarti fungsi valid
        except Exception:
            # jika terjadi error saat fungsi dijalankan
            return False
            # berarti fungsi tidak valid pada titik tersebut

    @staticmethod
    def turunan_numerik(fungsi, x, h=1e-6):
        """
        Menghitung turunan numerik fungsi.
        (Jika turunan tidak diberikan)
        """
        return (fungsi(x + h) - fungsi(x - h)) / (2 * h)
        # menggunakan metode beda hingga (central difference)
        # rumus pendekatan turunan:
         # f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    @classmethod
    def dari_fungsi(cls, fungsi, x0, toleransi=1e-6, maks_iter=50):
        """
        Membuat objek hanya dari fungsi tanpa turunan eksplisit.
        (Turunan dihitung otomatis)
        """
        turunan = lambda x: cls.turunan_numerik(fungsi, x)
        # membuat fungsi turunan otomatis menggunakan lambda
        # memanggil method turunan_numerik

        return cls(
            fungsi=fungsi,
            turunan=turunan,
            x0=x0,
            toleransi=toleransi,
            maks_iter=maks_iter
        )
         # mengembalikan objek NewtonRaphson baru

    def __repr__(self):
        """
        Representasi objek saat dicetak.
        """
        return (
            f"NewtonRaphson("
            f"x0={self.x0}, "
            f"toleransi={self.toleransi}, "
            f"maks_iter={self.maks_iter})"
        )
        # menentukan tampilan objek saat dipanggil dengan print()
