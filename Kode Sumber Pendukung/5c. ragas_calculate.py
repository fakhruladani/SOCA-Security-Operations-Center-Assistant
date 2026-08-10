# ============================================================
# 5c. ragas_calculate.py — Tahap 3/3 pipeline evaluasi RAGAS: Scoring
# ============================================================
# Alur: baca question+answer+contexts+ground_truth dari checkpoint (hasil 5a+5b)
# -> bungkus 1 sampel jadi 1-row HuggingFace Dataset -> ragas.evaluate() dengan
# LLM judge gpt-4o-mini (GitHub Models) untuk 4 metrik: faithfulness,
# answer_relevancy, context_precision, context_recall -> simpan skor ke
# checkpoint -> di akhir, rekap rata-rata (keseluruhan + per kategori) ke
# hasil_ragas_evaluation.xlsx.
# Dievaluasi SATU SAMPEL PER PANGGILAN (bukan batch) supaya progres bisa
# disimpan incremental dan resumable kalau rate limit/error di tengah jalan.
# Resumable: sampel yang 4 skornya sudah lengkap di-skip; jawaban [Error:]
# (gagal generate di 5b) di-skip juga, bukan dinilai sebagai skor 0.
# Prasyarat: jalankan "5a. ragas_retrieval.py" + "5b. ragas_generation.py".
# Jalankan: python "5c. ragas_calculate.py"
# ============================================================

import os
import time
import json
import math
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent

METRICS     = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRICS_OBJ = [faithfulness, answer_relevancy, context_precision, context_recall]
CATS        = ["log_based", "mitre_attack", "nist_csf"]

# Jeda antar sampel agar tidak kena rate limit GitHub Models (~15 RPM)
# Tiap sampel ~10-15 API call → 90 detik (dinaikkan dari 60 karena masih timeout).
RAGAS_SLEEP = 90


def _safe_score(val):
    """Konversi hasil skor RAGAS ke float bersih; NaN atau tipe tak valid -> None.
    RAGAS bisa mengembalikan NaN kalau metrik gagal dihitung (mis. context kosong)."""
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _all_scores_done(entry: dict) -> bool:
    """True kalau ke-4 metrik sudah terisi (bukan None) -> entry di-skip, tidak dinilai ulang."""
    return all(entry.get(m) is not None for m in METRICS)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
CHECKPOINT = _DIR / "ragas_final_checkpoint.json"
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        "ragas_final_checkpoint.json tidak ditemukan.\n"
        "Jalankan '5a. ragas_retrieval.py' dan '5b. ragas_generation.py' terlebih dahulu."
    )

checkpoint_map = {e["id"]: e for e in json.loads(CHECKPOINT.read_text(encoding="utf-8"))}
logger.info("Checkpoint dimuat: %d entry.", len(checkpoint_map))


def _save():
    """Tulis ulang seluruh checkpoint_map ke ragas_final_checkpoint.json (dipanggil
    tiap 1 sampel selesai dinilai — supaya skor tidak hilang kalau rate limit/error)."""
    CHECKPOINT.write_text(
        json.dumps(list(checkpoint_map.values()), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# Cek jawaban error
error_entries = [e for e in checkpoint_map.values()
                 if str(e.get("answer", "")).startswith("[Error:")]
if error_entries:
    logger.warning(
        "%d jawaban masih [Error:] — di-skip. Jalankan '5b. ragas_generation.py' untuk memperbaiki.",
        len(error_entries)
    )

# Sampel yang perlu dievaluasi
need_eval = [
    e for e in checkpoint_map.values()
    if not _all_scores_done(e) and not str(e.get("answer", "")).startswith("[Error:")
]

logger.info("=" * 62)
logger.info("Perlu evaluasi   : %d sampel", len(need_eval))
logger.info("Sudah selesai    : %d sampel", sum(1 for e in checkpoint_map.values() if _all_scores_done(e)))
logger.info("Jawaban error    : %d sampel (di-skip)", len(error_entries))
logger.info("=" * 62)

if not need_eval:
    logger.info("Semua skor sudah lengkap. Lanjut ke output Excel.")
else:
    RAGAS_KEY = os.environ.get("RAGAS_API_KEY", "")
    if not RAGAS_KEY:
        raise EnvironmentError("RAGAS_API_KEY belum diisi di .env")

    # temperature=0 penting untuk justifikasi metodologi: hasil LLM judge jadi
    # deterministik, sehingga evaluasi cukup dijalankan SATU KALI (bukan berulang
    # untuk rata-rata). Model juri diatur via .env agar bebas provider (RAGAS_MODEL,
    # RAGAS_BASE_URL, RAGAS_API_KEY). Catatan: layanan GitHub Models yang dipakai
    # saat penelitian telah pensiun; default kini mengarah ke OpenAI.
    ragas_llm = ChatOpenAI(
        model           = os.getenv("RAGAS_MODEL", "gpt-4o-mini"),
        api_key         = RAGAS_KEY,
        base_url        = os.getenv("RAGAS_BASE_URL", "https://api.openai.com/v1"),
        temperature     = 0,
        request_timeout = 300,
        max_retries     = 5,
    )
    # Embedding judge HARUS sama dengan yang dipakai sistem live (SOCA.py) —
    # context_precision/recall RAGAS menghitung kemiripan semantik memakai
    # model ini, jadi kalau beda model, skornya tidak mencerminkan retrieval asli.
    ragas_embed = HuggingFaceEmbeddings(
        model_name    = "BAAI/bge-large-en-v1.5",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    for i, entry in enumerate(need_eval, 1):
        sid      = entry["id"]
        q        = entry["question"]
        answer   = entry["answer"]
        contexts = entry["contexts"]
        gnd      = entry["ground_truth"]
        cat      = entry["category"]

        logger.info("[%d/%d] [%s] %s", i, len(need_eval), cat, q[:70])

        # Dataset RAGAS berisi 1 baris saja per panggilan evaluate() — dinilai
        # satu-per-satu (bukan batch semua 30 sekaligus) supaya kalau macet di
        # tengah, sampel yang sudah dinilai tidak ikut hilang/diulang.
        single_ds = Dataset.from_list([{
            "question":     q,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": gnd,
        }])

        try:
            res    = evaluate(
                dataset          = single_ds,
                metrics          = METRICS_OBJ,
                llm              = ragas_llm,
                embeddings       = ragas_embed,
                raise_exceptions = False,   # metrik yang gagal -> NaN, bukan crash seluruh evaluasi
                run_config       = RunConfig(max_workers=1),  # 1 sampel, tidak perlu paralel
            )
            row_df = res.to_pandas()
            scores = {m: _safe_score(row_df[m].iloc[0]) if m in row_df.columns else None
                      for m in METRICS}
        except Exception as e:
            logger.warning("  RAGAS gagal: %s", e)
            scores = {m: None for m in METRICS}

        checkpoint_map[sid].update(scores)
        _save()

        done = sum(1 for e in checkpoint_map.values() if _all_scores_done(e))
        logger.info("  Skor    : %s",
                    {m: f"{scores[m]:.3f}" if scores[m] is not None else "null" for m in METRICS})
        logger.info("  Progress: %d/%d selesai.", done, len(checkpoint_map))

        if i < len(need_eval):
            logger.info("  Jeda %ds (rate limit)...", RAGAS_SLEEP)
            time.sleep(RAGAS_SLEEP)

logger.info("=== Evaluasi selesai. ===")

# ---------------------------------------------------------------------------
# Tampilkan hasil & simpan Excel
# ---------------------------------------------------------------------------
# df[m].mean() otomatis mengabaikan None/NaN (sampel error/belum dinilai)
# tanpa perlu filter manual — itu sebabnya skor rata-rata tetap valid meski
# masih ada sampel yang belum lengkap skornya.
df = pd.DataFrame(list(checkpoint_map.values()))

still_null  = sum(1 for e in checkpoint_map.values() if not _all_scores_done(e))
still_error = len(error_entries)

print("\n" + "=" * 62)
print("HASIL EVALUASI RAGAS — SOCA")
print("=" * 62)
print(f"\nTotal sampel  : {len(df)}")
print(f"Skor lengkap  : {len(df) - still_null}")
print(f"Masih null    : {still_null}")
print(f"Error (skip)  : {still_error}")

print("\nRata-rata keseluruhan:")
for m in METRICS:
    if m in df.columns:
        val = df[m].mean()
        print(f"  {m:<30}: {val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else f"  {m:<30}: -")

print("\nRata-rata per kategori:")
for cat in CATS:
    sub = df[df["category"] == cat]
    if not len(sub):
        continue
    print(f"\n  [{cat.upper()}]  n={len(sub)}")
    for m in METRICS:
        if m in sub.columns:
            val = sub[m].mean()
            print(f"    {m:<30}: {val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else f"    {m:<30}: -")

out = _DIR / "hasil_ragas_evaluation.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    # Sheet "Detail": seluruh 30 baris checkpoint mentah (untuk lampiran/audit).
    # Sheet "Ringkasan": tabel rata-rata siap-pakai untuk Tabel 4.4 di BAB IV.
    df.to_excel(writer, sheet_name="Detail", index=False)

    summary = []
    for m in METRICS:
        if m not in df.columns:
            continue
        row = {"Metrik": m, f"Semua (n={len(df)})": round(df[m].mean(), 4)}
        for cat in CATS:
            sub = df[df["category"] == cat]
            row[cat] = round(sub[m].mean(), 4) if len(sub) else "-"
        summary.append(row)
    pd.DataFrame(summary).to_excel(writer, sheet_name="Ringkasan", index=False)

logger.info("Hasil disimpan ke: %s", out)
print(f"\nHasil disimpan ke: {out}")
