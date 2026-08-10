# ============================================================
# 5a. ragas_retrieval.py — Tahap 1/3 pipeline evaluasi RAGAS: Retrieval
# ============================================================
# Alur: import "SOCA.py" sebagai modul -> untuk tiap pertanyaan ground
# truth.json, panggil classify_intent() + retrieve_context() + format_
# contexts_for_eval() — fungsi ASLI dari SOCA.py, BUKAN logika duplikat —
# supaya RAGAS benar-benar mengukur perilaku sistem live, bukan simulasi
# terpisah. Hasil (field "contexts") disimpan ke ragas_final_checkpoint.json.
# Resumable: entry yang sudah punya contexts (bukan dataset-only/kosong)
# di-skip, jadi aman dijalankan ulang parsial. Urutan pipeline penuh:
#   5a (retrieval) -> 5b (generation) -> 5c (hitung skor RAGAS)
# Jalankan: python "5a. ragas_retrieval.py"
# ============================================================

import json
import logging
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Import pipeline SOCA
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("app", _DIR.parent / "SOCA.py")
_app  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)
_app.read_state_from_qdrant()

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# Format konteks & keputusan retrieval SEPENUHNYA memakai fungsi SOCA.py
# (classify_intent, retrieve_context, format_contexts_for_eval) agar RAGAS
# benar-benar mengukur perilaku sistem SOCA.py — bukan logika duplikat.


# ---------------------------------------------------------------------------
# Load ground truth & checkpoint
# ---------------------------------------------------------------------------
gt_path = _DIR / "ground truth.json"
if not gt_path.exists():
    raise FileNotFoundError("File 'ground truth.json' tidak ditemukan.")

ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
logger.info("Ground truth: %d pertanyaan", len(ground_truth))

CHECKPOINT = _DIR / "ragas_final_checkpoint.json"
checkpoint_map = {}
if CHECKPOINT.exists():
    checkpoint_map = {e["id"]: e for e in json.loads(CHECKPOINT.read_text(encoding="utf-8"))}
    logger.info("Checkpoint ditemukan: %d entry.", len(checkpoint_map))


def _save():
    """Tulis ulang seluruh checkpoint_map ke ragas_final_checkpoint.json (dipanggil
    setiap 1 entry selesai, bukan di akhir — supaya progres tidak hilang kalau macet)."""
    CHECKPOINT.write_text(
        json.dumps(list(checkpoint_map.values()), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Helper: deteksi entry yang hanya punya konteks dataset (belum di-retrieve KB)
# ---------------------------------------------------------------------------
def _only_dataset_or_empty(entry: dict) -> bool:
    """True kalau entry BELUM punya retrieval nyata (kosong, placeholder
    "(tidak ada konteks)", atau cuma baris [Dataset] metadata) -> perlu di-retrieve ulang."""
    ctx = entry.get("contexts") or []
    if len(ctx) == 0:
        return True
    if ctx == ["(tidak ada konteks)"]:
        return True
    # Hanya berisi satu string metadata dataset, tanpa log/KB sama sekali
    if all(c.startswith("[Dataset]") for c in ctx):
        return True
    return False


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
need = [item for item in ground_truth
        if _only_dataset_or_empty(checkpoint_map.get(item["id"], {}))]
logger.info("Perlu retrieval: %d/%d", len(need), len(ground_truth))

for i, item in enumerate(need, 1):
    q   = item["question"]
    sid = item["id"]
    cat = item["category"]
    gnd = item["ground_truth"]
    logger.info("[%d/%d] [%s] %s", i, len(need), cat, q[:75])

    # Keputusan retrieval SEPENUHNYA dari SOCA.py (classify_intent) — sama persis
    # dengan yang dipakai handler live, sehingga RAGAS mengukur sistem SOCA.py.
    intent = _app.classify_intent(q)
    logger.info("  intent → logs=%s nist=%s mitre=%s (tech=%s know=%s log=%s)",
                intent["needs_logs"], intent["needs_nist"], intent["needs_mitre"],
                intent["technical"], intent["knowledge"], intent["log_analysis"])

    try:
        ctx = _app.retrieve_context(
            q,
            needs_logs  = intent["needs_logs"],
            needs_nist  = intent["needs_nist"],
            needs_mitre = intent["needs_mitre"],
        )
    except Exception as e:
        logger.warning("  Retrieval error: %s", e)
        ctx = {"logs": [], "nist": [], "mitre": []}

    contexts = _app.format_contexts_for_eval(ctx, q)
    logger.info("  %d log | %d nist | %d mitre → %d string",
                len(ctx.get("logs", [])), len(ctx.get("nist", [])),
                len(ctx.get("mitre", [])), len(contexts))

    existing = checkpoint_map.get(sid, {})
    checkpoint_map[sid] = {
        "id":           sid,
        "category":     cat,
        "question":     q,
        "ground_truth": gnd,
        "contexts":     contexts,
        "answer":       existing.get("answer", None),
        **{m: existing.get(m, None) for m in METRICS},
    }
    _save()
    logger.info("  [Checkpoint] Disimpan (%d/%d).", i, len(need))

logger.info("=== 5a selesai. Jalankan '5b. ragas_generation.py' ===")
