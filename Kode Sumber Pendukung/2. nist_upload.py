# ============================================================
# 2. nist_upload.py — Upload knowledge base NIST CSF 2.0 ke Qdrant
# ============================================================
# Alur: csf2.xlsx (sheet "CSF 2.0") -> parse_nist_csf_excel() (forward-fill
# merged cells function/category, pecah "PR.AA-01: <desc>" jadi sub_id+desc,
# buang subcategory [Withdrawn] peninggalan CSF v1.1) -> 1 subcategory aktif
# = 1 chunk teks -> embed (BAAI/bge-large-en-v1.5) -> upsert ke collection
# "nist_csf" di Qdrant. Full refresh: collection lama di-drop lalu dibuat
# ulang tiap dijalankan (beda dengan wazuh_logs yang upsert incremental).
# Jalankan sekali saat setup, atau ulang kalau csf2.xlsx diperbarui:
#   python "2. nist_upload.py" [path/ke/csf2.xlsx]
# ============================================================

import os
import sys
import uuid
import logging
import warnings

from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nist_upload")

# ============================================================
# KONFIGURASI
# Secret (QDRANT_API_KEY) dibaca dari .env — lihat .env.example
# ============================================================
load_dotenv()

QDRANT_HOST    = os.getenv("QDRANT_HOST", "")            # WAJIB: IP/hostname server Qdrant (isi di .env)
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")         # isi bila Qdrant butuh auth
COLLECTION     = "nist_csf"
VECTOR_SIZE    = 1024
BATCH_SIZE     = 16

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
NIST_FILE   = "csf2.xlsx"


# ============================================================
# PARSER — NIST CSF 2.0 Excel
# ============================================================
def parse_nist_csf_excel(filepath: str) -> list[dict]:
    """
    Parse sheet "CSF 2.0" jadi 1 chunk per subcategory aktif.
    Excel punya merged cells untuk function/category (ffill mengisi baris di
    bawahnya), dan subcategory berformat "PR.AA-01: <deskripsi>" yang dipecah
    jadi sub_id + description. Subcategory berlabel "[Withdrawn...]" (peninggalan
    CSF v1.1 yang sudah dicabut) DIBUANG — dari csf2.xlsx menghasilkan 106 chunk
    aktif (terverifikasi cocok dengan jumlah vector di collection nist_csf).
    """
    df = pd.read_excel(filepath, sheet_name="CSF 2.0", header=0)

    # Validasi jumlah kolom minimal
    if df.shape[1] < 5:
        raise ValueError(
            f"File Excel harus memiliki minimal 5 kolom, ditemukan {df.shape[1]}. "
            "Pastikan sheet 'CSF 2.0' berisi kolom: function, category, subcategory, "
            "implementation_examples, informative_references."
        )

    # Ganti nama kolom berdasarkan URUTAN, bukan nama header asli di Excel —
    # header resmi NIST kadang beda kapitalisasi/spasi antar rilis file, jadi
    # lebih aman berasumsi 5 kolom pertama SELALU dalam urutan tetap ini.
    df.columns = [
        "function", "category", "subcategory",
        "implementation_examples", "informative_references",
        *df.columns[5:],  # kolom tambahan jika ada (diabaikan)
    ]
    df = df[["function", "category", "subcategory",
             "implementation_examples", "informative_references"]]

    # Forward-fill kolom yang di-merge di Excel
    df["function"] = df["function"].ffill()
    df["category"] = df["category"].ffill()

    chunks = []
    skipped_withdrawn = 0

    for _, row in df.iterrows():
        subcategory = str(row["subcategory"]).strip()

        # Skip header rows dan baris kosong
        if subcategory in ("nan", "", "Subcategory"):
            continue

        # Skip withdrawn subcategories (warisan CSF v1.1 yang sudah ditarik)
        if "[Withdrawn" in subcategory or "[withdrawn" in subcategory:
            skipped_withdrawn += 1
            continue

        function   = str(row["function"]).strip()
        category   = str(row["category"]).strip()
        examples   = str(row["implementation_examples"]).strip()
        references = str(row["informative_references"]).strip()

        # str(NaN) menghasilkan literal string "nan" (bukan None/""), jadi
        # sel Excel kosong harus dicek sebagai string "nan" secara eksplisit.
        if examples   == "nan": examples   = ""
        if references == "nan": references = ""

        # Pisahkan sub_id dari deskripsi: "PR.AA-01: ..." → "PR.AA-01"
        if ":" in subcategory:
            sub_id      = subcategory.split(":")[0].strip()
            description = subcategory.split(":", 1)[1].strip()
        else:
            sub_id      = ""
            description = subcategory

        # Ambil kode singkat function dari dalam tanda kurung:
        # "PROTECT (PR): ..." → "PR"
        # "GOVERN (GV): ..."  → "GV"
        if "(" in function and ")" in function:
            open_idx   = function.index("(")
            close_idx  = function.index(")")
            function_short = function[open_idx + 1: close_idx].strip()
        else:
            function_short = ""

        # Truncate informative_references agar tidak mendominasi embedding
        refs_truncated = (references[:300] + "...") if len(references) > 300 else references

        text = (
            f"NIST CSF 2.0 | {sub_id}\n"
            f"Function: {function}\n"
            f"Category: {category}\n"
            f"Guideline: {description}\n"
            f"Recommended Actions:\n{examples}\n"
            f"Informative References: {refs_truncated}"
        ).strip()

        chunks.append({
            "id":             str(uuid.uuid4()),
            "text":           text,
            "source":         "nist_csf",
            "sub_id":         sub_id,
            "function":       function,
            "function_short": function_short,   # GV / ID / PR / DE / RS / RC
            "category":       category,
        })

    logger.info("Withdrawn subcategories dilewati: %d", skipped_withdrawn)
    return chunks


# ============================================================
# INDEX KE QDRANT
# ============================================================
def index_chunks(chunks: list[dict], qdrant: QdrantClient,
                 embedder: SentenceTransformer):
    """Embed teks tiap chunk (BAAI/bge-large-en-v1.5) lalu upsert per batch ke Qdrant."""
    logger.info("Indexing %d chunks ke Qdrant (batch=%d)...",
                len(chunks), BATCH_SIZE)

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="NIST CSF 2.0"):
        batch      = chunks[i : i + BATCH_SIZE]
        texts      = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)

        points = []
        for chunk, vector in zip(batch, embeddings):
            payload = {k: v for k, v in chunk.items() if k != "id"}
            points.append(PointStruct(
                id=chunk["id"],
                vector=vector.tolist(),
                payload=payload,
            ))
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)


# ============================================================
# MAIN
# ============================================================
def main(filepath: str = NIST_FILE):
    """Entry point: drop+recreate collection nist_csf (full refresh), parse
    csf2.xlsx, lalu index seluruh chunk. Jalankan: python "2. nist_upload.py"."""
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting ke Qdrant %s:%s", QDRANT_HOST, QDRANT_PORT)
    qdrant = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT,
        api_key=QDRANT_API_KEY, https=False, timeout=60,
    )

    # Drop & recreate (full refresh) — beda dengan wazuh_logs yang di-upsert
    # incremental. KB NIST tidak pernah "bertambah sedikit-sedikit" seperti log,
    # jadi lebih sederhana & aman mulai dari collection kosong tiap dijalankan
    # daripada menangani deteksi entry usang/berubah.
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
        logger.info("Collection lama '%s' dihapus.", COLLECTION)

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Collection '%s' dibuat.", COLLECTION)

    logger.info("Parsing %s...", filepath)
    chunks = parse_nist_csf_excel(filepath)
    logger.info("Total NIST chunks aktif: %d", len(chunks))

    # Statistik distribusi per function
    dist: dict[str, int] = {}
    for c in chunks:
        fn = c["function_short"]
        dist[fn] = dist.get(fn, 0) + 1
    logger.info("Distribusi per function: %s", dist)

    index_chunks(chunks, qdrant, embedder)

    info = qdrant.get_collection(COLLECTION)
    logger.info("Selesai. Total vectors di Qdrant '%s': %d",
                COLLECTION, info.points_count)


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else NIST_FILE
    main(filepath)
