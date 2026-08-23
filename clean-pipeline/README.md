# Ambiguity-aware multi-expert pipeline

`pipeline.py` mereproduksi satu recipe dari pretrained weights publik dan data
resmi sampai `submission_final01.csv`. Nama eksperimen historis tidak menjadi
bagian metode. Implementasi teknis tetap dipisah agar setiap model dapat diaudit,
tetapi laporan memperlakukannya sebagai satu arsitektur end-to-end.

Pipeline tidak membaca checkpoint, probabilitas, submission, atau label test
dari eksperimen lama. `GT-FINAL.csv` dan ground truth lokal tidak digunakan.

## Hipotesis

Tiga kelas membutuhkan bukti visual berbeda. Recyclable banyak ditentukan oleh
material dan bentuk kemasan; Organic sering memerlukan konteks semantik; sedangkan
Electronic memiliki komponen dan geometri khas. Karena itu satu classifier global
digabung dengan expert terbatas untuk batas Recyclable–Organic.

Penambahan expert tidak didasarkan pada hasil test. Dasarnya adalah confusion
matrix dan probabilitas OOF train resmi: setelah global Macro-F1 tinggi, residual
error masih terkonsentrasi pada kelas `0` dan `2`. Kelas `1` tidak boleh diubah oleh
decision layer khusus ini.

## Arsitektur laporan

```text
Data train resmi
    ↓
Group-aware 5-fold protocol
    ├── Global visual anchor
    │   ├── material encoder
    │   ├── shape encoder
    │   └── semantic/context encoder
    ├── Class-boundary experts
    │   ├── hard-negative 0-vs-2 expert
    │   ├── neighborhood consistency
    │   ├── multiview stability
    │   ├── hierarchical SigLIP2 expert
    │   └── object-centric Patch-MIL expert
    ↓
OOF probabilities seluruh cabang
    ↓
Cross-fitted guarded router
    ↓
Training final pada seluruh train resmi
    ↓
Test inference dan submission
```

### 1. Official data protocol

- Memvalidasi 26.527 gambar train, label `0/1/2`, dan 1.458 ID test.
- Hash konten dipakai sebagai group identifier.
- `StratifiedGroupKFold`, lima fold, seed `2026` mencegah duplikat identik
  berada pada train dan validation sekaligus.
- Test hanya dibaca sebagai gambar tanpa label.

### 2. Global visual anchor

Anchor menggabungkan representasi komplementer:

- ConvNeXtV2-L 224 px: tekstur, material, tepi, dan bentuk lokal.
- DINOv3–ConvNeXt-L 224 px: bentuk dan semantik visual umum.
- Context encoder: konteks objek dan hubungan foreground–background.
- High-resolution context encoder: detail objek kecil dan gambar padat.

Augmentasi meliputi crop terkontrol, flip horizontal, perubahan ringan warna dan
cahaya, serta resize-pad yang mempertahankan aspect ratio. Probability fusion
dipilih dari OOF Macro-F1, bukan dari label test.

### 3. Class-boundary experts

Expert hanya menangani residual ambiguity `0 vs 2`:

- Hard-negative expert memberi bobot lebih besar pada contoh kelas `0` yang dekat
  dengan keputusan kelas `2`.
- Directional specialist mempelajari arah koreksi `2 → 0` tanpa mengubah seluruh
  decision surface.
- Neighborhood expert memeriksa apakah keputusan konsisten dengan tetangga pada
  embedding train.
- Multiview expert membandingkan full image, horizontal flip, center crop, dan
  corner crop untuk menghindari koreksi akibat satu view tidak stabil.
- Hierarchical SigLIP2 memakai shared encoder dengan head tiga kelas dan head
  biner `0 vs 2`, sehingga bukti global dan batas pasangan kelas tetap konsisten.
- Patch-MIL SigLIP2 menggabungkan prediksi global dan patch penting untuk gambar
  yang memuat beberapa objek atau background dominan.

Expert bukan pengganti anchor. Expert hanya memberikan bukti tambahan pada sampel
yang sudah dinilai ambigu oleh anchor.

### 4. Cross-fitted decision layer

Seluruh fitur router untuk data validation berasal dari prediksi out-of-fold.
Logistic router dilatih pada empat fold dan memprediksi fold tersisa. Proses
diulang lima kali sehingga target row tidak pernah dipakai melatih prediksinya
sendiri.

Boundary threshold dipilih otomatis dari grid validation. Kandidat diterima hanya
jika:

- OOF Macro-F1 meningkat terhadap anchor;
- mean Macro-F1 antar-fold tidak menurun;
- minimal empat dari lima fold tidak memburuk;
- F1 kelas Electronic tidak berubah;
- perubahan hanya memiliki arah `2 → 0`.

Tie-break memilih perubahan lebih sedikit. Tidak ada threshold yang dikunci agar
menyerupai hasil lokal. Bila tidak ada kandidat yang lolos, pipeline memakai
anchor tanpa koreksi tersebut.

## Mengapa expert tetap diperlukan ketika validation sudah tinggi?

Macro-F1 agregat tinggi tidak berarti semua batas kelas selesai. Confusion matrix
OOF menunjukkan jenis kesalahan, bukan hanya jumlahnya. Expert ditambahkan karena
residual error memiliki pola terarah dan dapat diuji pada fold yang tidak dipakai
melatih expert. Ablation yang perlu dilaporkan:

| Konfigurasi | Pertanyaan |
|---|---|
| Global anchor | Seberapa kuat classifier umum? |
| + hard-negative expert | Apakah batas `0 vs 2` membaik? |
| + neighborhood/multiview evidence | Apakah koreksi stabil terhadap representasi dan view? |
| + cross-fitted router | Apakah koreksi menambah Macro-F1 tanpa collateral error? |

Komponen hanya dipertahankan jika ablation OOF menjawab pertanyaan tersebut secara
positif. Hasil GT test lokal tidak dimasukkan ke tabel utama laporan.

## Hierarchical + Patch-MIL residual consensus

Pilot pada 5.308 held-out row train resmi meningkatkan Macro-F1 anchor dari
`0.987886` menjadi `0.988332`, dengan error `72 → 69` dan seluruh lima subset
guard tidak menurun. Test GT tidak digunakan. Berdasarkan bukti tersebut, kedua
expert menjadi komponen aktif dengan ketentuan lebih ketat: pipeline final melatih
lima fold expert penuh dan memilih confidence gate ulang dari OOF hasil retrain.

Gate hanya mengubah `2 → 0` ketika kedua expert memprediksi `0`. Jika hasil full
OOF tidak meningkatkan anchor atau gagal fold guard, pipeline otomatis memakai
anchor. Angka threshold pilot tidak di-hardcode.

## Reproduksibilitas

- Seed global: `2026`.
- Revision pretrained model dipin di source.
- Deterministic PyTorch/CUDA diaktifkan sejauh didukung kernel.
- Source aktif dipin SHA-256 pada `pipeline_manifest.json`.
- Pipeline state memungkinkan resume tanpa mengulang tahap selesai.
- Output akhir divalidasi memiliki kolom `id,predicted`, 1.458 row, ID unik, dan
  label hanya `0/1/2`.

Bit-for-bit lintas versi driver CUDA tidak dijamin, tetapi recipe, split, dan
keputusan model deterministik pada stack yang sama.

## Menjalankan pipeline

Prasyarat Modal: Volume `bdc2026-data`, Volume `bdc2026-model-cache`, dan Secret
`huggingface-secret`.

```powershell
python FINAL/01/clean/pipeline.py --profile honeylim3sour
```

Resume setelah koneksi terputus:

```powershell
python FINAL/01/clean/pipeline.py --profile honeylim3sour --resume
```

Audit DAG tanpa GPU:

```powershell
python FINAL/01/clean/pipeline.py --profile honeylim3sour --dry-run
```

Default hanya mengunduh artefak ringan. Tambahkan `--download-models` saat
checkpoint final memang diperlukan.

## Inference dari artefak sendiri

```powershell
python FINAL/01/clean/inference.py `
  --artifact-root FINAL/01/clean/workspace/rebuild `
  --output-dir FINAL/01/clean/outputs/artifact_inference
```

Inference ini mem-fit ulang decision layer secara deterministik dari probabilitas
OOF dan test yang dihasilkan pipeline sendiri. Opsi `--reference` hanya regression
test output, bukan bagian pemilihan model.
