# SOCA — Security Operations Center Assistant

Chatbot analisis log keamanan berbasis **LLM + RAG** untuk tim SOC PUSDATIN
Kementerian Pertanian. SOCA menerima pertanyaan analis dalam bahasa alami,
mengambil konteks dari log Wazuh dan knowledge base (MITRE ATT&CK, NIST CSF 2.0)
yang tersimpan di Qdrant, lalu menghasilkan analisis serta rekomendasi mitigasi
menggunakan LLM.

Paket ini berisi **kode chatbot (produk)** SOCA, ditambah subfolder
**`Kode Sumber Pendukung/`** untuk penyiapan data dan evaluasi (lihat bagian
"Kode Sumber Pendukung" di bawah).

> **Tutorial penggunaan** tersedia langsung di dalam aplikasi: buka menu
> **Panduan** setelah menjalankan SOCA dan login.

---

## Arsitektur Singkat

Alur satu permintaan:

```
Browser (UI React)
  -> WebSocket /ws/chat  (autentikasi via pesan pertama)
     -> guardrails_input()    : deteksi prompt injection (input)
     -> classify_intent()     : tentukan jenis kueri + koleksi yang diperlukan
     -> retrieve_context()    : ambil log + referensi NIST/MITRE dari Qdrant
     -> build_*_prompt()      : susun prompt sesuai jenis kueri
     -> generate_answer()     : panggil LLM (Qwen3-8B via LM Studio)
     -> clean_output()        : bersihkan format keluaran
     -> guardrails_output()   : deteksi prompt injection (output)
  -> kirim jawaban ke UI
```

Log Wazuh diambil dari VPS via SSH (`/reload`, `/sync`, auto-sync 30 detik) dan
diindeks ke Qdrant sebagai vektor embedding.

---

## Stack Teknologi

| Komponen  | Implementasi                                     |
|-----------|--------------------------------------------------|
| LLM       | Qwen3-8B via LM Studio lokal (`qwen/qwen3-8b`)   |
| Embedding | `BAAI/bge-large-en-v1.5` (1024 dimensi, cosine)  |
| Vector DB | Qdrant (koleksi `wazuh_logs`, `nist_csf`, `mitre_attack`) |
| Safety    | Guardrails AI, 2 lapis (regex lokal + LLM judge) |
| Backend   | FastAPI + Uvicorn, port 8000                     |
| Frontend  | React 18 via CDN (tanpa npm)                     |

---

## Struktur Folder

```
Kode Sumber/
  SOCA.py                    Backend chatbot: FastAPI + seluruh pipeline RAG
  ui/index.html              Antarmuka web (React via CDN)
  requirements.txt           Dependensi Python (chatbot)
  .env.example               Template konfigurasi (salin jadi .env)
  README.md                  Berkas ini
  Kode Sumber Pendukung/     Skrip penyiapan data & evaluasi RAGAS (README tersendiri)
```

---

## Prasyarat

- **Python 3.13**
- **LM Studio** dengan model `qwen/qwen3-8b` di-Load, server aktif di
  `http://127.0.0.1:1234`
- **Qdrant** yang sudah berisi koleksi `wazuh_logs`, `nist_csf`, `mitre_attack`
- File **`.env`** (salin dari `.env.example`, isi kredensialnya)

---

## Cara Menjalankan

1. Pasang dependensi (di virtual environment disarankan):
   ```bash
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

2. Salin konfigurasi lalu isi nilainya:
   ```bash
   cp .env.example .env
   ```

3. Nyalakan LM Studio: Load `qwen/qwen3-8b`, lalu Start Server (port 1234).

4. (Sekali saja) daftarkan host key VPS sebelum `/reload` pertama:
   ```bash
   ssh-keyscan -H <ip_vps_wazuh> >> known_hosts
   ```

5. Jalankan backend:
   ```bash
   python SOCA.py
   ```

6. Buka browser ke `http://localhost:8000`, login dengan
   `APP_USERNAME` / `APP_PASSWORD` dari `.env`.

7. Setelah login, buka menu **Panduan** untuk **tutorial penggunaan lengkap**
   (contoh pertanyaan, daftar perintah, dan cara membaca jawaban).

---

## Konfigurasi (`.env`)

**Semua nilai deployment (alamat server, port, kredensial, endpoint LLM) dibaca
dari `.env`.** Tidak ada alamat yang di-hardcode di dalam kode, jadi kamu tinggal
menyalin `.env.example` menjadi `.env` lalu mengisinya sesuai lingkunganmu sendiri.
Kolom "Wajib" menandai nilai yang harus diisi.

| Variabel             | Wajib | Keterangan                                             |
|----------------------|-------|--------------------------------------------------------|
| `QDRANT_HOST`        | ya    | IP/hostname server Qdrant                              |
| `QDRANT_PORT`        | -     | Port Qdrant (default `6333`)                           |
| `QDRANT_API_KEY`     | -     | API key Qdrant (kosongkan bila tanpa auth)             |
| `QDRANT_HTTPS`       | -     | `true`/`false` (default `false`)                       |
| `LLM_BASE_URL`       | -     | Endpoint LLM (default LM Studio `http://127.0.0.1:1234/v1`) |
| `LLM_API_KEY`        | -     | Key LLM (default `lm-studio`, dummy untuk LM Studio)   |
| `LLM_MODEL`          | -     | Nama model LLM (default `qwen/qwen3-8b`)               |
| `SSH_HOST`           | -*    | IP/hostname server Wazuh                               |
| `SSH_PORT`           | -     | Port SSH (default `22`)                                |
| `SSH_USER`           | -     | User SSH (default `root`)                              |
| `SSH_PASSWORD`       | -*    | Password SSH                                           |
| `ARCHIVES_PATH`      | -     | Path arsip log Wazuh (default `/var/ossec/logs/archives/archives.json`) |
| `APP_USERNAME`       | -     | Username login UI (default `admin`)                    |
| `APP_PASSWORD`       | ya    | Password login UI                                      |
| `GUARDRAILS_MODEL`   | -     | Model juri Guardrails Lapis 2 (litellm, default `openai/gpt-4o-mini`) |
| `GUARDRAILS_API_KEY` | -     | API key model juri; kosong = fallback ke Lapis 1 (regex) |

*) `SSH_HOST` dan `SSH_PASSWORD` hanya wajib bila memakai fitur `/reload` dan
auto-sync (menarik log dari server Wazuh). Tanpa itu, chatbot tetap berjalan
memakai data yang sudah ada di Qdrant.

Parameter RAG (nama koleksi, `TOP_K`, model embedding) diatur sebagai konstanta
di bagian atas `SOCA.py`.

---

## Perintah Chatbot (WebSocket)

| Perintah         | Fungsi                                               |
|------------------|------------------------------------------------------|
| `/help`          | Menampilkan daftar perintah                          |
| `/stat`          | Status sistem (jumlah log, model, auto-sync, Guardrails) |
| `/reload`        | Tarik log dari VPS lalu indeks ke Qdrant             |
| `/sync`          | Sinkronisasi log baru sejak sinkronisasi terakhir    |
| `/full_analyze`  | Analisis komprehensif seluruh log                    |
| `/clear_logs`    | Hapus semua log dari Qdrant                          |
| `/clear_chat`    | Hapus riwayat percakapan sesi ini                    |
| `/set days N`    | Ubah rentang log ke N hari (1 sampai 365)            |

> Daftar dan contoh penggunaan perintah ini juga dijelaskan pada halaman
> **Panduan** di dalam aplikasi.

---

## Kode Sumber Pendukung

Paket ini juga menyertakan subfolder **`Kode Sumber Pendukung/`** berisi skrip di
luar chatbot: pipeline **penyiapan data** (mengisi knowledge base MITRE/NIST dan
log ke Qdrant) serta pipeline **evaluasi RAGAS** (mengukur kualitas jawaban). Skrip
ini opsional untuk sekadar menjalankan chatbot, tetapi diperlukan bila kamu ingin
membangun knowledge base dari nol atau mereproduksi evaluasi.

Penjelasan lengkap (daftar skrip, urutan menjalankan, dan prasyarat) ada di
**`Kode Sumber Pendukung/README.md`**.

---

## Catatan

- LLM final yang didokumentasikan pada Tugas Akhir adalah **Qwen3-8B via LM Studio**.
- Guardrails Lapis 2 (LLM-as-judge) bersifat provider-agnostic lewat litellm:
  atur `GUARDRAILS_MODEL` dan `GUARDRAILS_API_KEY` untuk memakai OpenAI, Anthropic,
  Gemini, model lokal, dan lainnya. Bila tidak dikonfigurasi, sistem otomatis
  memakai Lapis 1 (regex lokal).
- Skrip penyiapan knowledge base dan evaluasi (RAGAS) disertakan di subfolder
  `Kode Sumber Pendukung/` (lihat README di dalamnya).
