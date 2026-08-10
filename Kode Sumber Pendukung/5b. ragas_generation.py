# ============================================================
# 5b. ragas_generation.py — Tahap 2/3 pipeline evaluasi RAGAS: Generation
# ============================================================
# Alur: baca "contexts" hasil 5a dari checkpoint -> untuk tiap entry yang
# belum punya jawaban valid, panggil classify_intent() + build_*_prompt() +
# generate_answer() + clean_output() — fungsi ASLI SOCA.py, LLM live (Qwen3-8B
# via LM Studio) — supaya jawaban yang dinilai RAGAS persis jawaban yang akan
# dilihat analis SOC sungguhan. Hasil (field "answer") disimpan ke checkpoint.
# Auto-retry: kalau context terlalu panjang (overflow token LLM lokal),
# ulangi dengan TOP_K_LOGS/RECENT dikecilkan ke 3 dan max_tokens diturunkan.
# Resumable: jawaban yang sudah valid (bukan error/kependekan/terpotong)
# di-skip. Urutan pipeline penuh:
#   5a (retrieval) -> 5b (generation) -> 5c (hitung skor RAGAS)
# Jalankan: python "5b. ragas_generation.py"
# ============================================================

import re
import json
import logging
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI as _OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ---------------------------------------------------------------------------
# Import pipeline SOCA
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("app", _DIR.parent / "SOCA.py")
_app  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)

_app.llm = _OpenAI(base_url=_app.LLM_BASE_URL, api_key=_app.LLM_API_KEY, timeout=None)
_app.read_state_from_qdrant()


def _generate(prompt: str, is_technical: bool, max_tokens: int) -> str:
    """Panggil generate_answer() ASLI SOCA.py (sudah menyertakan /no_think Qwen3
    + max_tokens anti-truncation) lalu clean_output() — persis pipeline handler
    live, sehingga RAGAS mengukur sistem SOCA.py, bukan logika duplikat."""
    raw = _app.generate_answer(prompt, is_technical, max_tokens=max_tokens)
    return _app.clean_output(raw)


def build_prompt(q: str, ctx: dict, intent: dict) -> tuple[str, bool]:
    """Pilih build_*_prompt() ASLI SOCA.py — urutan kondisi SAMA PERSIS dengan
    handler live (lihat classify_intent), supaya prompt yang dievaluasi RAGAS
    identik dengan yang dikirim ke LLM saat chat sungguhan."""
    is_tech = intent["technical"]
    if intent["knowledge"] and not intent["log_analysis"]:
        return _app.build_knowledge_prompt(q, ctx, []), is_tech
    elif is_tech:
        return _app.build_technical_prompt(q, ctx, [], is_knowledge=intent["knowledge"]), is_tech
    else:
        return _app.build_casual_prompt(q, []), is_tech


def retrieve(q: str, intent: dict) -> dict:
    """Retrieval ulang (bukan pakai contexts tersimpan dari 5a) — perlu objek
    hit Qdrant asli (bukan string) untuk dibentuk ulang jadi prompt via
    build_*_prompt(). Keputusan koleksi dari SOCA.py.classify_intent (sama
    dengan 5a & live)."""
    return _app.retrieve_context(
        q,
        needs_logs  = intent["needs_logs"],
        needs_nist  = intent["needs_nist"],
        needs_mitre = intent["needs_mitre"],
    )


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
CHECKPOINT = _DIR / "ragas_final_checkpoint.json"
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        "ragas_final_checkpoint.json tidak ditemukan.\n"
        "Jalankan '5a. ragas_retrieval.py' terlebih dahulu."
    )

checkpoint_map = {e["id"]: e for e in json.loads(CHECKPOINT.read_text(encoding="utf-8"))}
logger.info("Checkpoint dimuat: %d entry.", len(checkpoint_map))


def _save():
    """Tulis ulang seluruh checkpoint_map ke ragas_final_checkpoint.json (dipanggil
    tiap 1 entry selesai generate — supaya progres tidak hilang kalau LLM macet)."""
    CHECKPOINT.write_text(
        json.dumps(list(checkpoint_map.values()), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# Qwen3 lokal kadang membocorkan awal kalimat reasoning tanpa tag <think> resmi
# (mis. langsung mulai "Okay, let me analyze..."). Pola ini menangkap ciri-ciri
# kalimat pembuka reasoning gaya bahasa Inggris supaya _needs_generate() bisa
# mendeteksinya dan minta LLM generate ulang.
_RAW_THINK_RE = re.compile(
    r"^(Okay[,.]?\s|Let me |Let's |Alright[,.]?\s|Sure[,.]?\s|I need to |I'll |First[,]?\s|"
    r"Now[,]?\s|Looking at|Based on the|I'll analyze|I need to analyze|The user is asking|"
    r"Let me analyze|Looking through|I see that|I'll look)",
    re.IGNORECASE
)

# Karakter valid untuk mengakhiri kalimat
_SENTENCE_ENDERS = frozenset('.!?)"。')

def _needs_generate(entry: dict) -> bool:
    """True kalau jawaban tersimpan masih 'rusak' dan perlu di-generate ulang:
    kosong/error, mengandung sisa reasoning <think>, terlalu pendek (<50 char),
    atau terpotong (tidak diakhiri tanda baca kalimat)."""
    ans = entry.get("answer") or ""
    if not ans or ans.startswith("[Error:"):
        return True
    if "<think>" in ans:                    # thinking block tidak tersaring
        return True
    stripped = ans.strip()
    if _RAW_THINK_RE.match(stripped):       # thinking tanpa tag (Qwen3 terkadang langsung tulis)
        return True
    if len(stripped) < 50:                  # terlalu pendek
        return True
    # Jawaban terpotong: tidak diakhiri tanda baca kalimat
    if stripped[-1] not in _SENTENCE_ENDERS:
        return True
    return False


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
need = [e for e in checkpoint_map.values() if _needs_generate(e)]
logger.info("Perlu generate: %d/%d", len(need), len(checkpoint_map))

for i, entry in enumerate(need, 1):
    sid = entry["id"]
    q   = entry["question"]
    cat = entry["category"]
    logger.info("[%d/%d] [%s] %s", i, len(need), cat, q[:75])

    # Klasifikasi intent via SOCA.py (sama dengan 5a & handler live)
    intent = _app.classify_intent(q)

    # Coba normal dulu
    try:
        ctx             = retrieve(q, intent)
        prompt, is_tech = build_prompt(q, ctx, intent)
        answer          = _generate(prompt, is_tech, max_tokens=1200 if is_tech else 1000)
        logger.info("  OK: %d karakter", len(answer))
    except Exception as e:
        err_str = str(e)
        logger.warning("  Error: %s", err_str[:120])

        # Retry dengan context lebih pendek jika overflow
        if "overflow" in err_str.lower() or "context" in err_str.lower() or "400" in err_str:
            logger.info("  Retry dengan TOP_K=3 dan context pendek...")
            try:
                _app.TOP_K_LOGS   = 3
                _app.TOP_K_RECENT = 3
                ctx             = retrieve(q, intent)
                prompt, is_tech = build_prompt(q, ctx, intent)
                new_contexts    = _app.format_contexts_for_eval(ctx, q)
                answer          = _generate(prompt, is_tech, max_tokens=700 if is_tech else 600)
                checkpoint_map[sid]["contexts"] = new_contexts
                logger.info("  Retry OK: %d karakter", len(answer))
            except Exception as e2:
                logger.warning("  Masih error: %s", str(e2)[:80])
                answer = f"[Error: {e2}]"
            finally:
                _app.TOP_K_LOGS   = 8
                _app.TOP_K_RECENT = 8
        else:
            answer = f"[Error: {e}]"

    checkpoint_map[sid]["answer"] = answer
    for m in METRICS:
        checkpoint_map[sid][m] = None  # reset skor agar 5c evaluasi ulang
    _save()
    logger.info("  [Checkpoint] Disimpan (%d/%d).", i, len(need))

remaining_errors = sum(1 for e in checkpoint_map.values() if _needs_generate(e))
if remaining_errors:
    logger.warning("%d jawaban masih error. Coba restart LM Studio lalu jalankan ulang.", remaining_errors)
else:
    logger.info("=== 5b selesai. Jalankan '5c. ragas_calculate.py' ===")
