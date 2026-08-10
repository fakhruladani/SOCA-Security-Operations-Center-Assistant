# Kode Sumber Pendukung — SOCA

Folder ini berisi skrip pendukung SOCA di luar chatbot utama (`../SOCA.py`). Skrip
di sini tidak melayani pengguna secara langsung, melainkan dijalankan **sekali saat
setup** (menyiapkan data) atau **saat pengujian** (evaluasi RAGAS).

Ada dua kelompok:

## A. Pipeline Penyiapan Data (mengisi Qdrant)

Jalankan berurutan. Tujuannya mengisi knowledge base dan log ke Qdrant agar chatbot
punya bahan untuk RAG.

| Skrip | Fungsi | Input yang kamu sediakan |
|-------|--------|--------------------------|
| `1. mitre_upload.py` | Index MITRE ATT&CK ke Qdrant (koleksi `mitre_attack`) | `mitre_attack.json` |
| `2. nist_upload.py` | Index NIST CSF 2.0 ke Qdrant (koleksi `nist_csf`) | `csf2.xlsx` |
| `3. convert_qradar_to_wazuh.py` | Konversi CSV ekspor QRadar menjadi JSONL bergaya Wazuh | CSV ekspor QRadar |
| `4. insert_qradar_logs_to_qdrant.py` | Embed & index log ke Qdrant (koleksi `wazuh_logs`) | JSONL hasil skrip 3 |

## B. Pipeline Evaluasi RAGAS (mengukur kualitas jawaban)

Jalankan berurutan setelah Qdrant terisi. Checkpoint-based dan resumable (kalau
berhenti, jalankan lagi maka lanjut).

| Skrip | Fungsi |
|-------|--------|
| `5a. ragas_retrieval.py` | Ambil konteks untuk tiap pertanyaan (mengimpor `../SOCA.py`) |
| `5b. ragas_generation.py` | Hasilkan jawaban via LLM (mengimpor `../SOCA.py`) |
| `5c. ragas_calculate.py` | Hitung skor RAGAS (LLM-as-judge) dan tulis rekap Excel |

---

## Prasyarat

1. **`.env` di folder induk** (`../.env`). Skrip di sini membaca `.env` yang **sama**
   dengan chatbot. Lihat `../.env.example`. Untuk evaluasi (`5c`) diperlukan juga
   `RAGAS_MODEL`, `RAGAS_BASE_URL`, `RAGAS_API_KEY`.

2. **Dependensi** (di virtual environment yang sama):
   ```bash
   pip install -r ../requirements.txt                                   # dependensi chatbot
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt                                      # tambahan folder ini
   ```

3. **Qdrant aktif** (alamat diatur di `.env`).

4. **LM Studio aktif** untuk `5a`/`5b`, karena keduanya mengimpor `../SOCA.py`
   (yang memakai LLM lokal). Untuk `5c`, juri RAGAS diatur via `.env`.

5. **Data yang kamu sediakan sendiri:** `mitre_attack.json`, `csf2.xlsx`, CSV ekspor
   QRadar, dan `ground truth.json` (dataset evaluasi berisi daftar pertanyaan +
   jawaban acuan). Letakkan sesuai yang dibaca tiap skrip (lihat komentar di
   masing-masing file).

---

## Cara Menjalankan

Jalankan dari dalam folder ini (`Kode Sumber Pendukung`).

### A. Menyiapkan knowledge base & log (sekali saat setup)
```bash
python "1. mitre_upload.py"
python "2. nist_upload.py"
python "3. convert_qradar_to_wazuh.py"
python "4. insert_qradar_logs_to_qdrant.py"
```

### B. Evaluasi RAGAS (butuh ground truth.json + Qdrant terisi + LLM aktif)
```bash
python "5a. ragas_retrieval.py"
python "5b. ragas_generation.py"
python "5c. ragas_calculate.py"
```
Hasil akhir: skor RAGAS (faithfulness, answer_relevancy, context_precision,
context_recall) beserta file Excel rekap.

---

## Catatan

- **Tanpa kredensial hardcoded.** Semua alamat server dan kunci dibaca dari `.env`
  (tidak ada nilai rahasia di dalam kode).
- **Struktur folder harus dipertahankan.** `5a`/`5b` mengimpor chatbot dari
  `../SOCA.py`, jadi folder ini harus tetap berada di dalam folder `Kode Sumber`.
- **Juri RAGAS bebas provider.** Default mengarah ke OpenAI. Layanan GitHub Models
  yang dipakai saat penelitian telah pensiun; ganti `RAGAS_MODEL` / `RAGAS_BASE_URL`
  di `.env` untuk memakai provider lain (mis. model lokal via endpoint OpenAI-compatible).
