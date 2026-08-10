# ============================================================
# SOCA (Security Operations Center Assistant) — backend utama
# ============================================================
# Alur request singkat (lihat tiap bagian bertanda "====" untuk detail):
#   1. Analis SOC kirim pertanyaan via WebSocket /ws/chat
#   2. guardrails_input()   -> validasi/blok prompt injection (regex)
#   3. classify_intent()    -> tentukan jenis pertanyaan (casual/knowledge/
#                              log_analysis/technical) & koleksi Qdrant yang perlu di-query
#   4. retrieve_context()   -> ambil log Wazuh + referensi NIST CSF/MITRE ATT&CK
#                              yang relevan dari Qdrant (vector search + exact ID match)
#   5. build_*_prompt()     -> susun prompt sesuai jenis pertanyaan
#   6. generate_answer()    -> panggil LLM (Qwen3-8B via LM Studio)
#   7. clean_output() + guardrails_output() -> bersihkan & validasi jawaban sebelum dikirim
# Log Wazuh diambil dari VPS via SSH (/reload, /sync, auto-sync 30 detik) dan
# diindeks ke Qdrant sebagai vector embedding (BAAI/bge-large-en-v1.5).
# ============================================================

import asyncio
import os
import re
import json
import uuid
import logging
import secrets
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()  # baca .env sebelum apapun diinisialisasi

# ← qdrant, openai, sentence_transformers HARUS di-load SEBELUM fastapi/uvicorn/paramiko
#   untuk menghindari konflik DLL PyTorch di Windows
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    OrderBy, Direction,
)
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import paramiko
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# GUARDRAILS AI — deteksi prompt injection
# ============================================================
_INJECTION_RE = re.compile(
    r"\b("
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)"
    r"|forget\s+(all\s+)?(previous|prior|what\s+you('?ve|\s+have)\s+(been\s+)?(told|said|instructed))"
    r"|disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|guidelines?)"
    r"|override\s+(your\s+)?(instructions?|rules?|guidelines?|system\s+prompt)"
    r"|you\s+are\s+now\s+(?!SOCA|the\s+SOC|an?\s+SOC)"
    r"|act\s+as\s+(if\s+)?(you\s+are\s+|a\s+)?(?!SOCA|SOC)"
    r"|pretend\s+(to\s+be|you\s+are)"
    r"|roleplay\s+as"
    r"|new\s+(instructions?|rules?|prompt|system\s+prompt)"
    r"|system\s+prompt\s*:"
    r"|<\s*system\s*>"
    r"|jangan\s+(ikuti|patuhi|lakukan)\s+instruksi"
    r"|abaikan\s+(semua\s+)?(instruksi|aturan|perintah)\s+(sebelumnya|di\s+atas)"
    r"|lupakan\s+(instruksi|aturan|perintah)"
    r"|sekarang\s+kamu\s+adalah"
    r"|berpura.pura\s+(menjadi|sebagai|kamu\s+adalah)"
    r"|do\s+anything\s+now"
    r"|without\s+(any\s+)?(restrictions?|limitations?|filters?|censorship)"
    r"|(disable|remove|turn\s+off|bypass)\s+(all\s+|any\s+)?(content\s+)?"
    r"(filters?|restrictions?|safety\s+(measures?|guidelines?)|guardrails?)"
    r"|system\s+update\s*:"
    r"|real\s+admin(istrator)?\s+(has\s+)?(taken\s+over|now\s+in\s+control)"
    r"|(previous|prior)\s+(user|message|session)\s+was\s+(a\s+)?(test|fake)\s+account"
    r"|end\s+of\s+(the\s+)?(user\s+)?(input|prompt|message)"
    r"|do(es)?\s+not\s+have\s+to\s+(abide\s+by|follow|obey)\s+(any\s+)?rules?"
    r"|(great\s+job|well\s+done)\b.{0,30}task\s+complete"
    r"|tanpa\s+(batasan|pembatasan|filter|sensor)\b"
    r"|(nonaktifkan|matikan|hapus)\s+(semua\s+)?filter"
    r"|admin\s+(asli|sebenarnya)\s+(sudah\s+)?(mengambil\s+alih|mengendalikan)"
    r"|akhir\s+dari\s+input\s+pengguna"
    r")",
    re.IGNORECASE,
)

try:
    from guardrails import Guard
    from guardrails.validator_base import Validator, register_validator, PassResult, FailResult
    from guardrails.errors import ValidationError as _GuardrailsValidationError

    @register_validator(name="soca-prompt-injection", data_type="string")
    class _PromptInjectionValidator(Validator):
        """Guardrails AI custom validator — deteksi prompt injection via regex lokal."""
        def validate(self, value, metadata={}):
            if _INJECTION_RE.search(str(value)):
                return FailResult(error_message="Prompt injection terdeteksi.")
            return PassResult()

    _guard_input = Guard().use(_PromptInjectionValidator(on_fail="exception"))
    _guardrails_available = True
    logging.getLogger("soca.boot").info(
        "Guardrails AI Lapis 1 (regex lokal) aktif."
    )
except Exception as _gr_init_err:
    _GuardrailsValidationError = Exception   
    _guard_input = None
    _guardrails_available = False
    logging.getLogger("soca.boot").warning(
        "Guardrails Lapis 1 gagal init: %s — fallback ke regex langsung.", _gr_init_err
    )

# Lapis 2 — LLM-as-judge (Guardrails Hub asli). Terpisah dari try/except Lapis 1
# di atas supaya kalau Lapis 2 gagal init (mis. API key belum diisi), Lapis 1
# (regex) tetap aktif — bukan ikut mati.
#
# Provider-agnostic lewat litellm: atur GUARDRAILS_MODEL (contoh
# "openai/gpt-4o-mini", "anthropic/claude-3-5-haiku-latest", "gemini/gemini-1.5-flash",
# "github/gpt-4o-mini", atau model lokal). GUARDRAILS_API_KEY diteruskan ke variabel
# env provider yang sesuai (mis. OPENAI_API_KEY) agar litellm membacanya.
GUARDRAILS_MODEL   = os.getenv("GUARDRAILS_MODEL", "openai/gpt-4o-mini")
GUARDRAILS_API_KEY = os.getenv("GUARDRAILS_API_KEY", "")
if GUARDRAILS_API_KEY and "/" in GUARDRAILS_MODEL:
    _gr_provider = GUARDRAILS_MODEL.split("/", 1)[0].upper()
    os.environ.setdefault(f"{_gr_provider}_API_KEY", GUARDRAILS_API_KEY)

try:
    from guardrails.hub import PromptInjectionDetector as _HubPromptInjectionDetector

    _guard_llm = Guard().use(
        _HubPromptInjectionDetector(
            llm_callable=GUARDRAILS_MODEL, threshold=0.8, on_fail="exception"
        )
    )
    _guardrails_llm_available = True
    logging.getLogger("soca.boot").info(
        "Guardrails AI Lapis 2 (LLM-as-judge, %s) aktif.", GUARDRAILS_MODEL
    )
except Exception as _gr_llm_init_err:
    _guard_llm = None
    _guardrails_llm_available = False
    logging.getLogger("soca.boot").warning(
        "Guardrails Lapis 2 (LLM) gagal init: %s — hanya Lapis 1 (regex) yang aktif.",
        _gr_llm_init_err,
    )

# ============================================================
# LOGGING
# Output ke dua tujuan sekaligus:
#   1. Terminal (StreamHandler) — seperti biasa
#   2. File soca_activity.log di direktori yang sama dengan file ini
#      (RotatingFileHandler — maks 5MB per file, simpan 3 file lama)
# ============================================================
import logging.handlers

_LOG_FILE = Path(__file__).parent / "soca_activity.log"
_LOG_FMT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE = "%Y-%m-%d %H:%M:%S"

# Root logger
_root = logging.getLogger()
_root.setLevel(logging.INFO)

# Handler 1 — Terminal
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE))
_root.addHandler(_console_handler)

# Handler 2 — File (append, rotate saat >5MB, simpan 3 backup)
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
    encoding="utf-8",
    mode="a",  # append — log lama tidak terhapus saat app restart
)
_file_handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE))
_root.addHandler(_file_handler)

logger = logging.getLogger("soca")
logger.info("Log file: %s", _LOG_FILE)

# Suppress log dari library eksternal yang tidak relevan
# paramiko: log SSH connect/auth di setiap sync — terlalu verbose
# httpx: log setiap HTTP request ke Qdrant
# sentence_transformers: log model loading
for _noisy in ("paramiko", "paramiko.transport", "httpx", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ============================================================
# KONFIGURASI
# Semua nilai deployment (host, port, kredensial, URL LLM) dibaca dari .env.
# Salin .env.example menjadi .env lalu isi sesuai lingkunganmu.
# ============================================================

# --- Qdrant (vector database) ---
QDRANT_HOST    = os.getenv("QDRANT_HOST", "")            # WAJIB: IP/hostname server Qdrant (isi di .env)
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")         # isi bila Qdrant butuh auth
QDRANT_HTTPS   = os.getenv("QDRANT_HTTPS", "false").lower() in ("true", "1", "yes")

COL_LOGS    = "wazuh_logs"
COL_NIST    = "nist_csf"
COL_MITRE   = "mitre_attack"
VECTOR_SIZE = 1024                     # harus cocok dengan dimensi EMBED_MODEL

# --- LLM chat (default LM Studio lokal; bisa diarahkan ke server lain via .env) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "lm-studio")     # LM Studio tidak butuh key asli
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen/qwen3-8b")   # model final TA: Qwen3-8B
LLM_TIMEOUT  = None

# --- Embedding (harus sama dengan yang dipakai saat indexing ke Qdrant) ---
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# --- SSH ke server Wazuh (untuk /reload & auto-sync menarik log) ---
SSH_HOST      = os.getenv("SSH_HOST", "")                # isi bila pakai /reload: IP/hostname server Wazuh
SSH_PORT      = int(os.getenv("SSH_PORT", "22"))
SSH_USER      = os.getenv("SSH_USER", "root")
SSH_PASSWORD  = os.getenv("SSH_PASSWORD", "")            # isi bila pakai /reload
ARCHIVES_PATH = os.getenv("ARCHIVES_PATH", "/var/ossec/logs/archives/archives.json")

# --- App auth (login UI) ---
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")             # WAJIB: password login UI

# --- RAG params ---
TOP_K_LOGS     = 3    # log diambil per query analisis
TOP_K_RECENT   = 8    # log diambil untuk query "terbaru/terakhir" (sort by timestamp DESC)
TOP_K_NIST     = 3    # naik dari 2 → cakupan referensi lebih baik (context_recall) tanpa banyak menurunkan precision
TOP_K_MITRE    = 3    # naik dari 2 → idem
EMBED_BATCH    = 64
UPSERT_BATCH   = 128
MIN_RULE_LEVEL = 5
DEFAULT_DAYS   = 7
MAX_LOGS_INDEX = 20000
ENDPOINT_ONLY  = True

# ============================================================
# INIT GLOBAL
# ============================================================
logger.info("Connecting ke Qdrant %s:%s", QDRANT_HOST, QDRANT_PORT)
try:
    qdrant = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=QDRANT_HTTPS,
        timeout=30,
        prefer_grpc=False,   # ← pakai REST HTTP, hindari konflik protobuf
    )
except Exception as e:
    logger.error("Gagal koneksi ke Qdrant: %s", e)
    raise

logger.info("Connecting ke LLM: %s (model=%s)", LLM_BASE_URL, LLM_MODEL)
try:
    llm = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=LLM_TIMEOUT,
    )
except Exception as e:
    logger.error("Gagal init LLM client: %s", e)
    raise

logger.info("Loading embedding model: %s", EMBED_MODEL)
try:
    embedder = SentenceTransformer(EMBED_MODEL)
except Exception as e:
    logger.error("Gagal memuat embedding model: %s", e)
    raise

# State aplikasi
app_state = {
    "days_range":      DEFAULT_DAYS,
    "logs_indexed":    0,
    "logs_metadata":   {"total": 0, "earliest": "-", "latest": "-"},
    "last_sync_ts":    None,   # datetime — timestamp log terakhir yang sudah diindex
    "auto_sync_on":    False,  # flag untuk enable/disable background sync
    "sync_interval":   30,     # detik antar setiap background sync
    "sync_in_progress": False, # cegah overlap jika sync sebelumnya belum selesai
}

# ============================================================
# AUTENTIKASI (HTTP & WebSocket)
# ============================================================
security = HTTPBasic()

def verify_credentials(username: str, password: str) -> bool:
    """Bandingkan username/password pakai compare_digest — tahan timing attack."""
    u_ok = secrets.compare_digest(username or "", APP_USERNAME)
    p_ok = secrets.compare_digest(password or "", APP_PASSWORD)
    return u_ok and p_ok

def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Dependency FastAPI untuk endpoint HTTP Basic Auth (tidak dipakai WebSocket —
    WebSocket autentikasi via pesan pertama, lihat websocket_endpoint)."""
    if not verify_credentials(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ============================================================
# GUARDRAILS AI
# ============================================================
# Perlindungan input menggunakan Guardrails AI — detector berjalan lokal (regex).

# Pesan tunggal untuk SEMUA deteksi prompt injection (Lapis 1 maupun Lapis 2).
# Detail lapis sengaja TIDAK ditampilkan ke pengguna — selain lebih rapi, ini
# mencegah penyerang menyimpulkan lapis mana yang menangkap/meloloskan input.
# Informasi lapis tetap dicatat di log server untuk audit & evaluasi.
_INJECTION_MSG = "Input terdeteksi mengandung upaya prompt injection."

# Toggle runtime Guardrails — dikendalikan perintah tersembunyi /guardrails on|off.
# Saat False, deteksi prompt injection (Lapis 1 & 2) DILEWATI pada input maupun
# output; validasi panjang tetap berjalan. Berguna untuk eksperimen dan menghemat
# panggilan API LLM judge. Default True (aman).
_guardrails_enabled = True


def _run_injection_guard(text: str, sisi: str) -> bool:
    """Deteksi prompt injection 2 lapis pada `text`. Return True jika TERDETEKSI.
    Dipakai bersama oleh guardrails_input() dan guardrails_output() (sesuai draf
    IV.2.4 — Guardrails berjalan di sisi input DAN output). `sisi` hanya untuk log.
      Lapis 1: regex lokal (_INJECTION_RE) — instan, gratis, deterministik.
      Lapis 2: LLM-as-judge (Guardrails Hub PromptInjectionDetector via litellm,
               model diatur GUARDRAILS_MODEL) — hanya dipanggil kalau Lapis 1 lolos."""
    # --- Lapis 1: regex lokal ---
    if _guardrails_available and _guard_input is not None:
        try:
            _guard_input.validate(text)
        except _GuardrailsValidationError:
            logger.warning("Prompt injection diblokir [%s] — Lapis 1 (regex).", sisi)
            return True
        except Exception as _api_err:
            # Error tak terduga dari framework Guardrails — validasi manual via regex
            logger.warning("Guardrails Lapis 1 error [%s] (fallback regex): %s — %s",
                           sisi, type(_api_err).__name__, _api_err)
            if _INJECTION_RE.search(text):
                logger.warning("Prompt injection diblokir [%s] — Lapis 1 (regex, fallback).", sisi)
                return True
    else:
        # Guardrails tidak terinisialisasi — pakai regex lokal langsung
        if _INJECTION_RE.search(text):
            logger.warning("Prompt injection diblokir [%s] — Lapis 1 (regex, Guardrails off).", sisi)
            return True

    # --- Lapis 2: LLM-as-judge (hanya dipanggil kalau Lapis 1 lolos) ---
    if _guardrails_llm_available and _guard_llm is not None:
        try:
            _guard_llm.validate(text)
        except _GuardrailsValidationError:
            logger.warning("Prompt injection diblokir [%s] — Lapis 2 (skor > threshold).", sisi)
            return True
        except RuntimeError as _rt_err:
            # Package membungkus semua error litellm (termasuk content-filter Azure) jadi
            # RuntimeError generik — dicek dari isi pesan. Content-filter = sinyal kuat
            # -> deteksi positif; selain itu error infrastruktur murni -> lolos (fail-open).
            _msg = str(_rt_err).lower()
            if "content management policy" in _msg or "filtered" in _msg:
                logger.warning("Prompt injection diblokir [%s] — Lapis 2 (content filter).", sisi)
                return True
            logger.warning("Guardrails Lapis 2 RuntimeError [%s], lanjut tanpa cek: %s", sisi, _rt_err)
        except TypeError:
            # BUG package: FailResult(errorMessage=...) camelCase -> TypeError setiap kali
            # respons judge bukan angka bersih. Konservatif = deteksi positif.
            logger.warning("Guardrails Lapis 2 TypeError bug [%s] (errorMessage vs "
                           "error_message) — deteksi positif konservatif.", sisi)
            return True
        except Exception as _llm_err:
            # Error infrastruktur murni (network/rate limit/auth) — jangan blok
            logger.warning("Guardrails Lapis 2 error [%s], lanjut tanpa cek: %s — %s",
                           sisi, type(_llm_err).__name__, _llm_err)
    return False

def guardrails_input(query: str) -> tuple[bool, str]:
    """
    Validasi kueri masukan sebelum masuk pipeline RAG (draf IV.2.4).
      1) Validasi panjang: 3–2000 karakter (SELALU aktif).
      2) Deteksi prompt injection 2 lapis via _run_injection_guard() — DILEWATI
         kalau Guardrails dinonaktifkan lewat perintah /guardrails off.
    Return (True, query) jika valid, (False, pesan_error) jika ditolak.
    """
    if not query or len(query.strip()) < 3:
        return False, "Query terlalu pendek."
    if len(query) > 2000:
        return False, "Query terlalu panjang (maks 2000 karakter)."

    if _guardrails_enabled and _run_injection_guard(query, "input"):
        return False, _INJECTION_MSG
    return True, query


def guardrails_output(response: str) -> tuple[bool, str]:
    """
    Validasi respons LLM sebelum dikirim ke analis (draf IV.2.4 — Guardrails juga
    berjalan di sisi output, bukan hanya input).
      1) Validasi panjang: minimal 10 karakter, potong bila > 8000 (SELALU aktif).
      2) Deteksi prompt injection via _run_injection_guard() pada respons —
         memastikan tidak ada instruksi tersembunyi/manipulasi yang tersisip di
         keluaran LLM. DILEWATI kalau Guardrails dinonaktifkan (/guardrails off).
    Return (True, response) jika valid, (False, pesan_error) jika ditolak.
    """
    if not response or len(response.strip()) < 10:
        return False, "Respons LLM kosong atau terlalu pendek."
    if len(response) > 8000:
        response = response[:8000] + "\n\n[Respons dipotong karena terlalu panjang]"

    if _guardrails_enabled and _run_injection_guard(response, "output"):
        return False, "Respons terdeteksi mengandung konten berbahaya."
    return True, response

# ============================================================
# SSH + LOAD LOGS
# ============================================================

# File known_hosts project-level — isi dengan:
#   ssh-keyscan -H <IP_SERVER_WAZUH> >> known_hosts
_KNOWN_HOSTS_FILE = Path(__file__).parent / "known_hosts"

def _make_ssh_client() -> paramiko.SSHClient:
    """Buka koneksi SSH ke VPS Wazuh (dipakai /reload dan /sync untuk baca archives.json)."""
    ssh = paramiko.SSHClient()
    try:
        ssh.load_system_host_keys()   # baca dari ~/.ssh/known_hosts
    except Exception:
        pass
    if _KNOWN_HOSTS_FILE.exists():
        try:
            ssh.load_host_keys(str(_KNOWN_HOSTS_FILE))
        except Exception:
            logger.warning("Gagal memuat %s", _KNOWN_HOSTS_FILE)

    # RejectPolicy: tolak host yang key-nya belum dikenal (cegah MITM)
    # Jika koneksi gagal karena host tidak dikenal, tambahkan key dengan:
    #   ssh-keyscan -H <IP_SERVER_WAZUH> >> known_hosts
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        ssh.connect(
            hostname=SSH_HOST,
            port=SSH_PORT,
            username=SSH_USER,
            password=SSH_PASSWORD,
            timeout=30,
        )
    except paramiko.SSHException as e:
        ssh.close()
        raise RuntimeError(
            f"Host key {SSH_HOST} tidak dikenal atau tidak cocok. "
            f"Jalankan: ssh-keyscan -H {SSH_HOST} >> known_hosts"
        ) from e
    return ssh


def load_logs_from_vps(past_days: int = 7) -> list[dict]:
    """
    Baca SELURUH archives.json dari VPS Wazuh (dipakai perintah /reload — full pull).
    Filter: level minimum (MIN_RULE_LEVEL), endpoint saja (buang log internal Wazuh
    manager), dan rentang hari (past_days). Beda dengan load_logs_since() yang
    hanya tail incremental untuk auto-sync/​/sync.
    """
    logger.info("Membaca archives.json dari VPS (%s)...", SSH_HOST)

    try:
        ssh = _make_ssh_client()
        _, stdout, _ = ssh.exec_command(f"cat {ARCHIVES_PATH}")
        raw_lines = stdout.readlines()
        ssh.close()
    except Exception as e:
        logger.error("SSH Error: %s", e)
        return []

    logger.info("Total baris dibaca: %d", len(raw_lines))

    cutoff  = datetime.now() - timedelta(days=past_days)
    logs    = []
    skipped = 0

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        # Filter level minimum
        try:
            rule_level = int(event.get("rule", {}).get("level", 0))
        except (TypeError, ValueError):
            rule_level = 0
        if rule_level < MIN_RULE_LEVEL:
            skipped += 1
            continue

        # Filter endpoint only — buang log internal Wazuh manager (agent.id == "000")
        # Konvensi Wazuh: agent.id "000" = manager itu sendiri (internal log)
        # agent.id "001" dst. = endpoint yang dipantau (log yang kita inginkan)
        if ENDPOINT_ONLY:
            agent_id = str(event.get("agent", {}).get("id", "")).strip()
            if agent_id == "000" or agent_id == "":
                skipped += 1
                continue

            # Filter tambahan: buang log dari lokasi internal Wazuh
            location = event.get("location", "")
            if any(loc in location for loc in (
                "ossec-monitord", "ossec-agentd", "ossec-analysisd",
                "wazuh-modulesd", "wazuh-db", "wazuh-execd",
                "/var/ossec/logs/",
            )):
                skipped += 1
                continue

        # Filter tanggal
        ts_str = event.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str[:19])
                if ts < cutoff:
                    skipped += 1
                    continue
            except ValueError:
                pass

        logs.append(event)
        if len(logs) >= MAX_LOGS_INDEX:
            logger.warning("Mencapai batas MAX_LOGS_INDEX=%d, sisa log di-skip.",
                           MAX_LOGS_INDEX)
            break

    # Update metadata
    timestamps = []
    for e in logs:
        ts = e.get("timestamp", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts[:19]))
            except ValueError:
                pass

    if timestamps:
        app_state["logs_metadata"] = {
            "total":    len(logs),
            "earliest": min(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
            "latest":   max(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        app_state["logs_metadata"] = {"total": len(logs), "earliest": "-", "latest": "-"}

    logger.info("Events valid: %d | Dilewati: %d", len(logs), skipped)
    return logs


def load_logs_since(since_ts: datetime, tail_lines: int = 2000) -> list[dict]:
    """
    Baca HANYA log baru sejak since_ts dari VPS menggunakan tail.
    Lebih efisien untuk background sync karena tidak baca seluruh archives.json —
    hanya mengambil N baris terakhir (append-only file, log baru selalu di bawah).
    tail_lines=2000 cukup untuk menampung log baru dalam interval 30 detik.
    """
    logger.debug("Sync incremental: ambil tail-%d baris, filter sejak %s",
                tail_lines, since_ts.isoformat())
    try:
        ssh = _make_ssh_client()
        _, stdout, _ = ssh.exec_command(
            f"tail -n {tail_lines} {ARCHIVES_PATH}"
        )
        raw_lines = stdout.readlines()
        ssh.close()
    except Exception as e:
        logger.error("SSH Error saat sync: %s", e)
        return []

    logs    = []
    skipped = 0

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        # Filter level minimum
        try:
            rule_level = int(event.get("rule", {}).get("level", 0))
        except (TypeError, ValueError):
            rule_level = 0
        if rule_level < MIN_RULE_LEVEL:
            skipped += 1
            continue

        # Filter endpoint only
        if ENDPOINT_ONLY:
            agent_id = str(event.get("agent", {}).get("id", "")).strip()
            if agent_id in ("000", ""):
                skipped += 1
                continue
            location = event.get("location", "")
            if any(loc in location for loc in (
                "ossec-monitord", "ossec-agentd", "ossec-analysisd",
                "wazuh-modulesd", "wazuh-db", "wazuh-execd",
                "/var/ossec/logs/",
            )):
                skipped += 1
                continue

        # Hanya ambil log yang lebih baru dari since_ts
        ts_str = event.get("timestamp", "")
        if not ts_str:
            skipped += 1
            continue
        try:
            ts = datetime.fromisoformat(ts_str[:19])
            if ts <= since_ts:
                skipped += 1
                continue
        except ValueError:
            skipped += 1
            continue

        logs.append(event)

    if logs:
        logger.info("Sync: %d log baru ditemukan dari %d baris tail (skip=%d)",
                    len(logs), len(raw_lines), skipped)
    else:
        logger.debug("Sync: tidak ada log baru dari %d baris tail (skip=%d)",
                     len(raw_lines), skipped)
    return logs

# ============================================================
# HELPERS
# ============================================================
def _join(v) -> str:
    """Gabungkan list jadi 'a, b, c' — dipakai untuk field payload Qdrant (mitre_id, groups, dll)."""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x)
    return str(v) if v else ""

# ------------------------------------------------------------
# Ekstraksi statistik flow dari RawLog CEF (Check Point/QRadar)
# ------------------------------------------------------------
# Field seperti dst=, bytes=, connection_count=, duration= sering muncul
# jauh di belakang string CEF (>800 karakter) — di luar jangkauan slice
# full_log yang dipotong untuk pembatasan panjang teks. Ekstrak di sini
# SEBELUM dipotong agar fakta kunci (IP tujuan, volume transfer, jumlah
# koneksi, durasi) tidak hilang dari data yang tersimpan/ditampilkan.
_FLOW_FIELD_PATTERNS = {
    "dst":         r"\bdst=([\d.]+)",
    "src":         r"\bsrc=([\d.]+)",
    "bytes":       r"\bbytes=(\d+)",
    "connections": r"\bconnection_count=(\d+)",
    "duration":    r"\bduration=(\d+)",
}

def _extract_flow_stats(full_log: str) -> dict:
    """Cari dst/src/bytes/connections/duration di RawLog CEF via regex, kembalikan dict
    berisi field yang ketemu saja (dict kosong kalau tidak ada satu pun cocok)."""
    if not full_log:
        return {}
    out = {}
    for key, pattern in _FLOW_FIELD_PATTERNS.items():
        m = re.search(pattern, full_log)
        if m:
            out[key] = m.group(1)
    return out

def _format_flow_stats(stats: dict) -> str:
    """Ringkasan kompak flow stats untuk ditambahkan ke prompt/konteks RAGAS."""
    if not stats:
        return ""
    parts = []
    if "src" in stats:
        parts.append(f"src={stats['src']}")
    if "dst" in stats:
        parts.append(f"dst={stats['dst']}")
    if "bytes" in stats:
        try:
            mb = int(stats["bytes"]) / (1024 * 1024)
            parts.append(f"bytes={stats['bytes']}(~{mb:.1f}MB)")
        except ValueError:
            parts.append(f"bytes={stats['bytes']}")
    if "connections" in stats:
        parts.append(f"connections={stats['connections']}")
    if "duration" in stats:
        try:
            jam = int(stats["duration"]) / 3600
            parts.append(f"duration={stats['duration']}s(~{jam:.1f}j)")
        except ValueError:
            parts.append(f"duration={stats['duration']}s")
    return " ".join(parts)

# ============================================================
# INDEX LOG WAZUH KE QDRANT
# ============================================================
def index_logs_to_qdrant(logs: list[dict]):
    """Upsert log baru ke collection wazuh_logs tanpa menghapus log lama.
    ID log bersifat deterministik (uuid5) sehingga log yang sama tidak diduplikat.
    """
    if not qdrant.collection_exists(COL_LOGS):
        qdrant.create_collection(
            collection_name=COL_LOGS,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    chunks = []
    for event in logs:
        timestamp = event.get("timestamp", "")
        date_str  = ""
        hour_str  = ""
        try:
            dt       = datetime.fromisoformat(timestamp[:19])
            date_str = dt.strftime("%Y-%m-%d")
            hour_str = dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            pass

        rule         = event.get("rule", {}) or {}
        rule_id      = str(rule.get("id", ""))
        rule_level   = int(rule.get("level", 0) or 0)
        rule_desc    = rule.get("description", "")
        rule_groups  = _join(rule.get("groups", []))
        try:
            rule_fired = int(rule.get("firedtimes", 1) or 1)
        except (TypeError, ValueError):
            rule_fired = 1

        mitre        = rule.get("mitre", {}) or {}
        mitre_id     = _join(mitre.get("id", []))
        mitre_tactic = _join(mitre.get("tactic", []))
        mitre_tech   = _join(mitre.get("technique", []))

        agent      = event.get("agent", {}) or {}
        agent_name = agent.get("name", "unknown")
        agent_id   = str(agent.get("id", ""))
        agent_ip   = agent.get("ip", "")

        # Pastikan hanya log dari endpoint (bukan internal Wazuh manager)
        if ENDPOINT_ONLY and agent_id.strip() in ("000", ""):
            continue

        data     = event.get("data", {}) or {}
        srcip    = data.get("srcip", "")
        srcport  = str(data.get("srcport", ""))
        srcuser  = data.get("srcuser", "")
        dstuser  = data.get("dstuser", "")
        protocol = data.get("protocol", "")
        url      = data.get("url", "")
        command  = data.get("command", "")

        location     = event.get("location", "")
        decoder      = event.get("decoder", {}) or {}
        decoder_name = decoder.get("name", "")
        full_log_raw = event.get("full_log", "") or ""
        flow_stats   = _extract_flow_stats(full_log_raw)
        full_log     = full_log_raw[:1500]

        pci_dss  = _join(rule.get("pci_dss", []))
        nist_800 = _join(rule.get("nist_800_53", []))

        flow_line = _format_flow_stats(flow_stats)
        text = (
            f"Wazuh Security Event\n"
            f"Tanggal: {date_str} Jam: {hour_str}\n"
            f"Agent: {agent_name} (IP: {agent_ip})\n"
            f"Rule {rule_id} Level {rule_level} Fired {rule_fired}x\n"
            f"Deskripsi: {rule_desc}\n"
            f"Groups: {rule_groups}\n"
            f"MITRE: {mitre_id} | Tactic: {mitre_tactic} | Technique: {mitre_tech}\n"
            f"Source IP: {srcip} Port: {srcport} User: {srcuser}\n"
            f"Dest User: {dstuser} Protocol: {protocol}\n"
            f"URL: {url}\n"
            f"Command: {command}\n"
            f"Location: {location} Decoder: {decoder_name}\n"
            + (f"Flow: {flow_line}\n" if flow_line else "")
            + f"Full Log: {full_log}"
        ).strip()

        payload = {
            "text":            text,
            "timestamp":       timestamp,
            "date":            date_str,
            "hour":            hour_str,
            "rule_id":         rule_id,
            "rule_level":      rule_level,
            "rule_desc":       rule_desc,
            "rule_groups":     rule_groups,
            "rule_fired":      rule_fired,
            "agent_name":      agent_name,
            "agent_id":        agent_id,
            "agent_ip":        agent_ip,
            "mitre_id":        mitre_id,
            "mitre_tactic":    mitre_tactic,
            "mitre_technique": mitre_tech,
            "srcip":           srcip,
            "srcport":         srcport,
            "srcuser":         srcuser,
            "dstuser":         dstuser,
            "protocol":        protocol,
            "url":             url,
            "command":         command,
            "location":        location,
            "decoder":         decoder_name,
            "full_log":        full_log,
            "flow_stats":      flow_stats,
            "pci_dss":         pci_dss,
            "nist_800_53":     nist_800,
            "source":          "wazuh_log",
        }

        # ID deterministik → log yang sama selalu dapat UUID yang sama, upsert tidak duplikat
        _key = f"{timestamp}|{agent_id}|{rule_id}|{full_log[:100]}"
        chunks.append({"id": str(uuid.uuid5(uuid.NAMESPACE_URL, _key)), "text": text, "payload": payload})

    logger.info("Indexing %d chunks ke Qdrant (batch=%d)...", len(chunks), EMBED_BATCH)

    import time
    import gc
    t0 = time.time()
    total_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH

    for i in range(0, len(chunks), EMBED_BATCH):
        batch        = chunks[i : i + EMBED_BATCH]
        texts        = [c["text"] for c in batch]
        embeddings   = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        points = [
            PointStruct(id=c["id"], vector=emb.tolist(), payload=c["payload"])
            for c, emb in zip(batch, embeddings)
        ]
        qdrant.upsert(collection_name=COL_LOGS, points=points, wait=True)

        batch_num = i // EMBED_BATCH + 1
        if batch_num % 10 == 0 or batch_num == total_batches:
            elapsed = time.time() - t0
            logger.info("  ...%d / %d chunks (batch %d/%d, %.1fs)",
                        min(i + EMBED_BATCH, len(chunks)), len(chunks),
                        batch_num, total_batches, elapsed)

    gc.collect()

    try:
        total = qdrant.get_collection(COL_LOGS).points_count or len(chunks)
    except Exception:
        total = len(chunks)
    app_state["logs_indexed"] = total

    # Update last_sync_ts ke timestamp log terbaru yang baru diindex
    new_timestamps = []
    for c in chunks:
        ts_str = c["payload"].get("timestamp", "")
        if ts_str:
            try:
                new_timestamps.append(datetime.fromisoformat(ts_str[:19]))
            except ValueError:
                pass
    if new_timestamps:
        latest_new = max(new_timestamps)
        current = app_state.get("last_sync_ts")
        if current is None or latest_new > current:
            app_state["last_sync_ts"] = latest_new
            logger.info("last_sync_ts diperbarui: %s", latest_new.isoformat())

    logger.info("Indexing Wazuh selesai. Batch: %d | Total di Qdrant: %d.", len(chunks), total)


def clear_logs_collection():
    """Hapus semua log dari Qdrant — drop + recreate collection wazuh_logs."""
    if qdrant.collection_exists(COL_LOGS):
        qdrant.delete_collection(COL_LOGS)
    qdrant.create_collection(
        collection_name=COL_LOGS,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    app_state["logs_indexed"] = 0
    app_state["logs_metadata"] = {"total": 0, "earliest": "-", "latest": "-"}
    logger.info("Collection '%s' dihapus dan dibuat ulang.", COL_LOGS)


# ============================================================
# RETRIEVAL (Hybrid: Exact ID match + Vector search)
# ============================================================

_MITRE_ID_RE = re.compile(r"\bT(\d{4})(?:\.(\d{3}))?\b", re.IGNORECASE)
_NIST_ID_RE  = re.compile(r"\b([A-Z]{2})\.([A-Z]{2})-(\d{2})\b")

def _extract_mitre_ids(query: str) -> list[str]:
    """Tangkap ID teknik MITRE (mis. T1110, T1557.002) yang disebut eksplisit di query."""
    ids = []
    for m in _MITRE_ID_RE.finditer(query):
        base, sub = m.group(1), m.group(2)
        ids.append(f"T{base}.{sub}" if sub else f"T{base}")
    return list(dict.fromkeys(ids))

def _extract_nist_ids(query: str) -> list[str]:
    """Tangkap ID subkategori NIST CSF (mis. DE.CM-01) yang disebut eksplisit di query."""
    return list(dict.fromkeys(m.group(0) for m in _NIST_ID_RE.finditer(query)))

def _hybrid_search(collection: str, query_vec: list, limit: int,
                   id_field: str = None, id_values: list = None,
                   min_level: int = None) -> list:
    """
    Retrieval "hybrid": gabungan exact-match ID (kalau query menyebut ID MITRE/NIST
    spesifik, prioritaskan entri itu — skor dipaksa >=0.99) + vector similarity search
    sebagai pelengkap untuk mengisi sisa kuota `limit`. Dedup by point id.
    """
    results  = []
    seen_ids = set()

    # Filter severity: buang log di bawah MIN_RULE_LEVEL saat retrieval (bukan hanya
    # saat ingestion) agar event kritis tidak tenggelam oleh noise level rendah.
    level_cond = ([FieldCondition(key="rule_level", range=Range(gte=min_level))]
                  if min_level is not None else [])

    if id_field and id_values:
        try:
            filter_obj = Filter(
                should=[
                    FieldCondition(key=id_field, match=MatchValue(value=v))
                    for v in id_values
                ],
                must=level_cond or None,
            )
            exact = qdrant.query_points(
                collection_name=collection,
                query=query_vec,
                query_filter=filter_obj,
                limit=limit,
                with_payload=True,
            ).points
            for p in exact:
                p.score = max(p.score, 0.99)
                results.append(p)
                seen_ids.add(p.id)
        except Exception as e:
            logger.warning("Exact-match filter error pada %s: %s", collection, e)

    if len(results) < limit:
        try:
            vec_hits = qdrant.query_points(
                collection_name=collection,
                query=query_vec,
                limit=limit * 2,
                query_filter=Filter(must=level_cond) if level_cond else None,
                with_payload=True,
            ).points
            for p in vec_hits:
                if p.id not in seen_ids:
                    results.append(p)
                    seen_ids.add(p.id)
                if len(results) >= limit:
                    break
        except Exception as e:
            logger.warning("Vector search error pada %s: %s", collection, e)

    return results[:limit]



def _fetch_recent_logs(limit: int = TOP_K_RECENT, min_level: int = None) -> list:
    """
    Ambil log terbaru diurutkan by timestamp DESC.
    Digunakan saat strategi retrieval 'recent'/'both' (query kejadian terkini).
    Berbeda dari _hybrid_search yang mengurutkan by semantic similarity.

    Pengurutan dilakukan di SISI PYTHON, bukan lewat order_by Qdrant: order_by
    Qdrant butuh payload index pada 'timestamp' yang tidak tersedia di koleksi
    wazuh_logs (Qdrant menolak dengan 400). Timestamp berformat ISO dengan zona
    waktu seragam (mis. 2026-04-21T08:52:05.000+0700), sehingga urutan
    leksikografis string = urutan kronologis — cukup di-sort biasa.
    """
    try:
        scroll_filter = (Filter(must=[FieldCondition(key="rule_level", range=Range(gte=min_level))])
                         if min_level is not None else None)
        # Scroll seluruh log yang lolos filter (paginasi), lalu sort di Python.
        collected, offset = [], None
        while True:
            points, offset = qdrant.scroll(
                collection_name=COL_LOGS,
                limit=512,
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(points)
            if offset is None or len(collected) >= MAX_LOGS_INDEX:
                break
        # Urutkan by timestamp menurun (terbaru dulu), ambil N teratas
        collected.sort(key=lambda p: (p.payload or {}).get("timestamp", ""), reverse=True)
        top = collected[:limit]
        for p in top:
            p.score = 1.0 
        logger.info("Recent logs fetched: %d dari %d (sort Python by timestamp DESC)",
                    len(top), len(collected))
        return top
    except Exception as e:
        logger.warning("_fetch_recent_logs error: %s — fallback ke vector search", e)
        return []


def _fetch_all_logs_scroll() -> list:
    """
    Ambil SEMUA log dari Qdrant via scroll tanpa filter vektor.
    Digunakan untuk analisis menyeluruh — tidak membatasi jumlah log.
    Hasil diurutkan timestamp DESC (terbaru di atas).
    """
    all_points = []
    offset = None
    try:
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=COL_LOGS,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None or len(points) == 0:
                break
            offset = next_offset
        all_points.sort(
            key=lambda p: (p.payload or {}).get("timestamp", ""),
            reverse=True,
        )
        logger.info("_fetch_all_logs_scroll: %d log total di Qdrant", len(all_points))
        return all_points
    except Exception as e:
        logger.warning("_fetch_all_logs_scroll error: %s", e)
        return []


# ============================================================
# KLASIFIKASI STRATEGI RETRIEVAL LOG
# ============================================================
_STRATEGY_SYS = (
    "Kamu mengklasifikasikan strategi pengambilan log untuk sistem analisis SOC. "
    "Jawab HANYA satu kata tanpa penjelasan: recent, search, atau both."
)

def _classify_retrieval(query: str) -> str:
    """Klasifikasi strategi retrieval log via panggilan LLM singkat (draf IV.2.2).
    LLM diminta menjawab satu kata:
      - recent : kejadian/log terkini berdasarkan waktu (mis. "serangan terbaru")
      - search : topik/jenis serangan tertentu (relevansi semantik)
      - both   : perlu keduanya (analisis menyeluruh, ringkasan, statistik)
    Retrieval sudah mendukung ketiganya (lihat retrieve_context). Fallback ke 'both'
    (pilihan aman) bila LLM gagal atau menjawab di luar format."""
    user = (
        "Tentukan strategi pengambilan log untuk pertanyaan analis berikut.\n"
        "recent = menanyakan kejadian atau log terbaru berdasarkan waktu.\n"
        "search = menanyakan topik atau jenis serangan tertentu.\n"
        "both = memerlukan keduanya (analisis menyeluruh, ringkasan, atau statistik).\n\n"
        f"Pertanyaan: {query}\n"
        "Jawab satu kata (recent/search/both):"
    )
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _STRATEGY_SYS + _NO_THINK},
                {"role": "user",   "content": user},
            ],
            temperature=0,
            max_tokens=8,
        )
        raw = (resp.choices[0].message.content or "").strip().lower().strip(".\"' ")
        decision = raw if raw in ("recent", "search", "both") else "both"
    except Exception as _e:
        # LLM gagal (mati/rate limit/format aneh) -> 'both' sebagai pilihan aman
        logger.warning("Klasifikasi strategi retrieval via LLM gagal (%s) — fallback 'both'.",
                       type(_e).__name__)
        decision = "both"
    logger.info("Retrieval classification: '%s' untuk query: %s", decision, query[:60])
    return decision


_COUNTING_RE = re.compile(
    r"\b(berapa|total|distribusi|semua|seluruh|keseluruhan|sebutkan|apa\s+saja|"
    r"siapa\s+saja|mana\s+saja|daftar|jumlah|berapa\s+banyak|berapa\s+event|berapa\s+log|"
    r"berapa\s+kali|berapa\s+insiden|berapa\s+koneksi|ringkasan|temuan)\b",
    re.IGNORECASE,
)

def _dynamic_top_k(query: str) -> int:
    """Query statistik/overview butuh lebih banyak sampel log dari query spesifik."""
    return 15 if _COUNTING_RE.search(query) else 8


# ------------------------------------------------------------------
# PENEKAN LOG KEMBAR (keragaman konteks)
# ------------------------------------------------------------------
# Audit koleksi wazuh_logs menemukan 356 titik nyaris-kembar: satu pola berulang
# sampai 147 kali (Check Point CEF, rule & source IP sama). ID-nya semua unik,
# jadi dedup berbasis point-id TIDAK menangkapnya. Akibatnya top-K bisa terisi
# beberapa log yang praktis identik, memboroskan jendela konteks dan membuat
# jawaban kehilangan keragaman bukti.
#
# Solusinya: kelompokkan per tanda tangan isi, ambil SATU wakil, dan simpan
# jumlah kejadian serupa ke payload (_dup_count) supaya informasi "ini terjadi
# N kali" tidak hilang — hanya barisnya yang tidak diulang.
# Faktor over-fetch ditetapkan dari pengukuran, bukan tebakan. Pada koleksi ini
# 8 log terbaru hanya berisi 2 kelompok unik, 16 log -> 2, 24 -> 7, 32 -> 9,
# 48 -> 10. Jadi 4x adalah ambang minimum untuk memenuhi target 8; dipakai 6x
# agar ada margin. Nyaris tanpa biaya tambahan karena _fetch_recent_logs memang
# sudah men-scroll seluruh koleksi lalu menyortir di sisi Python.
DEDUP_OVERFETCH = 6

def _log_signature(p: dict) -> tuple:
    """Tanda tangan isi sebuah log. Dua log dianggap kembar bila rule, sumber,
    tujuan, dan deskripsinya sama pada hari yang sama."""
    return (
        str(p.get("rule_id", "")),
        str(p.get("rule_desc", "")),
        str(p.get("srcip", "")),
        str(p.get("agent_name", "")),
        str(p.get("date", "")),
    )

def _dedup_log_hits(hits: list, target: int) -> tuple[list, int]:
    """Tekan log kembar lalu potong ke `target`. Return (hits, jumlah_ditekan).
    Wakil yang dipertahankan adalah yang PERTAMA muncul (urutan skor/waktu tetap
    dihormati). Payload wakil diberi '_dup_count'."""
    if not hits:
        return [], 0
    order, group = [], {}
    for h in hits:
        sig = _log_signature(h.payload or {})
        if sig in group:
            group[sig] += 1
        else:
            group[sig] = 1
            order.append((sig, h))
    for sig, h in order:
        if h.payload is not None:
            h.payload["_dup_count"] = group[sig]
    kept = [h for _, h in order][:target if target > 0 else len(order)]
    collapsed = len(hits) - len(kept)
    return kept, max(collapsed, 0)


# ============================================================
# RETRIEVAL ADAPTIF
# ============================================================
def retrieve_context(query: str,
                     needs_logs: bool = True,
                     needs_nist: bool = True,
                     needs_mitre: bool = True) -> dict:
    """
    Query 3 koleksi Qdrant (wazuh_logs, nist_csf, mitre_attack) sesuai kebutuhan
    (needs_logs/nist/mitre — ditentukan classify_intent() di pemanggil).
    Strategi retrieval log ditentukan oleh _classify_retrieval() (panggilan LLM singkat):
    "recent" -> _fetch_recent_logs (sort timestamp DESC), "search" -> _hybrid_search
    (vector similarity), "both" -> gabungan keduanya. needs_logs=False untuk knowledge
    query murni — hemat latency.
    """
    vector = embedder.encode(query, normalize_embeddings=True).tolist()

    mitre_ids = _extract_mitre_ids(query)
    nist_ids  = _extract_nist_ids(query)
    if mitre_ids:
        logger.info("Detected MITRE IDs in query: %s", mitre_ids)
    if nist_ids:
        logger.info("Detected NIST IDs in query: %s", nist_ids)

    log_hits = []
    if needs_logs and qdrant.collection_exists(COL_LOGS):
        strategy = _classify_retrieval(query)
        recent_hits = []
        vector_hits = []
        target = 0

        # Over-fetch (2x) lalu tekan log kembar, baru dipotong ke target. Tanpa
        # over-fetch, dedup justru MENGURANGI jumlah log yang sampai ke prompt.
        if strategy in ("recent", "both"):
            recent_hits = _fetch_recent_logs(TOP_K_RECENT * DEDUP_OVERFETCH,
                                             min_level=MIN_RULE_LEVEL)
            target += TOP_K_RECENT

        if strategy in ("search", "both"):
            top_k = _dynamic_top_k(query)
            vector_hits = _hybrid_search(
                COL_LOGS, vector, top_k * DEDUP_OVERFETCH,
                id_field="mitre_id", id_values=mitre_ids if mitre_ids else None,
                min_level=MIN_RULE_LEVEL,
            )
            target += top_k

        seen, merged = set(), []
        for p in recent_hits + vector_hits:
            if p.id not in seen:
                merged.append(p)
                seen.add(p.id)

        log_hits, collapsed = _dedup_log_hits(merged, target)
        logger.info("Log hits: strategy=%s | diambil=%d -> dipakai=%d "
                    "(%d log kembar ditekan)",
                    strategy, len(recent_hits) + len(vector_hits),
                    len(log_hits), collapsed)

    nist_hits = []
    if needs_nist and qdrant.collection_exists(COL_NIST):
        nist_hits = _hybrid_search(
            COL_NIST, vector, TOP_K_NIST,
            id_field="sub_id", id_values=nist_ids if nist_ids else None,
        )

    mitre_hits = []
    if needs_mitre and qdrant.collection_exists(COL_MITRE):
        mitre_hits = _hybrid_search(
            COL_MITRE, vector, TOP_K_MITRE,
            id_field="technique_id", id_values=mitre_ids if mitre_ids else None,
        )

    logger.info("Retrieved — log: %d, nist: %d, mitre: %d",
                len(log_hits), len(nist_hits), len(mitre_hits))
    return {"logs": log_hits, "nist": nist_hits, "mitre": mitre_hits}

# ============================================================
# PENENTUAN JENIS KUERI
# ============================================================
# 3 detektor regex independen (technical/knowledge/log_analysis) dikombinasikan
# di classify_intent() di bawah untuk menentukan: (a) perlu data log atau tidak,
# (b) perlu KB NIST/MITRE atau tidak, (c) prompt template mana yang dipakai.
# Semua berbasis regex — tidak ada panggilan LLM tambahan untuk klasifikasi ini.
TECHNICAL_PATTERNS = [
    r"\bbrute\s?force\b", r"\bsql\s?injection\b", r"\bxss\b", r"\bexploit\b",
    r"\bmalware\b", r"\bransomware\b", r"\bphishing\b", r"\bddos\b", r"\bnmap\b",
    r"\bbackdoor\b", r"\bwebshell\b", r"\brootkit\b", r"\bbotnet\b", r"\btrojan\b",
    r"\b(log|logs|alert|alerts|rule|level|wazuh|siem|event|agent|archives)\b",
    r"\b(deteksi|ancaman|serangan|insiden|incident|threat|attack|intrusion)\b",
    r"\b(ip|port|traffic|firewall|ufw|ssh|ftp|http|network|jaringan|koneksi)\b",
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    r"\b(mitre|att&?ck|nist|csf|taktik|teknik|tactic|technique|mitigasi|rekomendasi)\b",
    r"\bT\d{4}(?:\.\d{3})?\b",
    r"\b(analisa|analisis|investigasi|forensik|suspicious|mencurigakan|anomali)\b",
]
_TECH_RE = re.compile("|".join(TECHNICAL_PATTERNS), re.IGNORECASE)

def is_technical_query(query: str) -> bool:
    """True jika query menyinggung istilah teknis keamanan (log, IP, teknik MITRE, dll)."""
    return bool(_TECH_RE.search(query or ""))

KNOWLEDGE_PATTERNS = [
    # definisi / penjelasan konsep
    r"\b(apa|apakah)\s+(itu|yang\s+dimaksud|arti|maksud|pengertian)\b",
    r"\b(jelaskan|jelasin|definisikan|deskripsikan|terangkan)\b",
    r"\b(definisi|pengertian|maksud|arti)\s+(dari|tentang|mengenai)?\b",
    r"^apa\s+itu\b",
    # cara kerja / cara mengatasi / mitigasi (tangkap variasi "caranya", "cara untuk", dsb.)
    r"\bbagaimana\s+cara(nya)?\s+(kerja|mendeteksi|mencegah|mengatasi|menangani|melindungi|memitigasi|merespons|menghadapi)\b",
    r"\bcara\s+(mengatasi|mencegah|menangani|melindungi|memitigasi|merespons|menghadapi|mendeteksi)\b",
    r"\blangkah[\s\-]langkah\s+(untuk|dalam|mengatasi|mencegah|merespons|menangani)\b",
    r"\b(rekomendasi|mitigasi|solusi|penanganan|pencegahan)\s+(untuk|terhadap|dari|serangan|ancaman|insiden)?\b",
    r"\bbagaimana\s+(cara|langkah|strategi|prosedur)\b",
    r"\bapa\s+(yang\s+harus|yang\s+perlu|langkah|tindakan)\b",
    r"\b(harus|perlu|sebaiknya)\s+(dilakukan|diambil|diterapkan)\b",
    # ID spesifik MITRE / NIST
    r"\bT\d{4}(?:\.\d{3})?\b",
    r"\b(fungsi|kategori|subcategory)\s+(nist|csf)\b",
    r"\b[A-Z]{2}\.[A-Z]{2}-\d{2}\b",   # contoh DE.CM-01
    # klasifikasi / konsep
    r"\bmasuk\s+(kategori|taktik|teknik|jenis)\b",
    r"\btermasuk\s+(kategori|taktik|teknik|jenis|apa)\b",
    r"\bkategori\s+(apa|mitre|nist|att&?ck)\b",
    r"\b(contoh|ciri|karakteristik|indikator)\s+(dari|serangan|teknik|taktik)\b",
    r"\b(perbedaan|bedanya|perbandingan)\s+antara\b",
    # variasi "bagaimana" yang tidak butuh kata kunci "cara/langkah"
    r"\bbagaimana\s+(seharusnya|semestinya|sebaiknya|sebenarnya)\b",
    r"\bbagaimana\s+(kelima|keempat|ketiga|keenam)\b",
    r"\bbagaimana\s+(fungsi|peran|framework|model|konsep)\b",
    r"\bbagaimana\b.{0,40}\b(bekerja\s+bersama|saling\s+melengkapi|saling\s+mendukung)\b",
    # "bagaimana <subjek konsep> bekerja/mencegah/dst" — pertanyaan cara kerja teknik/kontrol
    r"\bbagaimana\s+(penyerang|penyusup|attacker|adversary|teknik|taktik|serangan|metode|kontrol|proses|mekanisme)\b",
    r"\bbagaimana\b.{0,60}\bbekerja\b",
    r"\bbagaimana\b.{0,50}\b(mencegah|melindungi|mengamankan|diterapkan|membantu)\b",
]
_KNOWLEDGE_RE = re.compile("|".join(KNOWLEDGE_PATTERNS), re.IGNORECASE)

def is_knowledge_query(query: str) -> bool:
    """True jika query menanyakan konsep/definisi/mitigasi (bukan analisis log nyata)."""
    return bool(_KNOWLEDGE_RE.search(query or ""))

# Pola yang membutuhkan analisis log nyata (bukan sekedar pengetahuan)
LOG_ANALYSIS_PATTERNS = [
    r"\b(analisis|analisa|investigasi)\s+(log|insiden|kejadian|serangan|alert)\b",
    r"\b(sebutkan|tampilkan|tunjukkan|lihat|cari)\s+.*(log|ip|alert|event|insiden)\b",
    r"\b(berapa|hitung|jumlah).*(log|event|alert|serangan|kali|percobaan)\b",
    r"\b(ip|alamat)\s+.*(mana|apa|berapa).*(serangan|attack|brute|mencurigakan)\b",
    r"\b(kapan|waktu|tanggal|jam).*(serangan|insiden|kejadian|terdeteksi)\b",
    r"\b(distribusi|frekuensi|tren|pola)\s+.*(event|log|serangan|alert)\b",
    r"\banalisis(lah)?\s+(log|semua|data)\b",
    r"\b(ringkasan|summary)\s+(log|insiden|event)\b",
    r"\blog\s+(yang\s+ada|wazuh|terbaru|terakhir)\b",
    r"\b(agent|endpoint)\s+(mana|apa)\s+.*(sering|banyak|tinggi)\b",
    # Pertanyaan yang menganalisis ISI dataset log (bukan sekadar konsep)
    r"\b(jelaskan|apakah\s+ada|adakah)\b.{0,60}\b(dalam|berdasarkan|dari|pada)\s+log\b",
    r"\b(terdeteksi|tercatat)\s+(dalam|di|pada)\s+log\b",
    r"\b(temuan|ringkasan)\b.{0,30}\blog\b",
    r"\bberdasarkan\s+log\b",
    r"\b(sumber|perangkat|sensor)\s+log\b",
    r"\blog\s+(keamanan|dalam\s+dataset)\b",
    r"\b(event|koneksi|aktivitas)\s+.{0,30}(mencurigakan|eksfiltrasi|paling)\b",
    r"\bhost\s+internal\s+mana\b",
]
_LOG_ANALYSIS_RE = re.compile("|".join(LOG_ANALYSIS_PATTERNS), re.IGNORECASE)

def is_log_analysis_query(query: str) -> bool:
    """True jika pertanyaan butuh data log nyata (bukan sekadar pengetahuan)."""
    return bool(_LOG_ANALYSIS_RE.search(query or ""))

# ------------------------------------------------------------
# ROUTING DOMAIN KB (MITRE vs NIST) untuk pertanyaan pengetahuan
# ------------------------------------------------------------
# Sinyal kuat MITRE ATT&CK: nama taktik/teknik & kosakata perilaku adversary.
_MITRE_DOMAIN_RE = re.compile(
    r"\b(mitre|att&?ck|teknik|taktik|technique|tactic|"
    r"command\s*(and|&)?\s*control|c2|lateral\s+movement|privilege\s+escalation|"
    r"credential\s+access|defense\s+evasion|initial\s+access|exfiltration|eksfiltrasi|"
    r"brute\s*force|adversary-?in-?the-?middle|reconnaissance|persistence|"
    r"penyerang|penyusup|attacker|adversary|menyalahgunakan)\b"
    r"|\bT\d{4}(?:\.\d{3})?\b",
    re.IGNORECASE,
)
# Sinyal kuat NIST CSF: nama fungsi & kosakata tata kelola/kontrol.
_NIST_DOMAIN_RE = re.compile(
    r"\b(nist|csf|framework|kerangka\s+kerja|govern(ance)?|tata\s+kelola|"
    r"continuous\s+monitoring|"
    r"fungsi\s+(detect|respond|identify|protect|recover|deteksi|respons|identifikasi|proteksi|pemulihan)|"
    r"kontrol\s+(keamanan|akses)|pengelolaan\s+identitas|manajemen\s+risiko|"
    r"kelima\s+fungsi|lima\s+fungsi|melindungi|perlindungan|"
    r"respons\s+(terhadap\s+)?insiden|recover)\b"
    r"|\b[A-Z]{2}\.[A-Z]{2}-\d{2}\b",
    re.IGNORECASE,
)

def detect_kb_domain(query: str) -> tuple[bool, bool]:
    """Tentukan KB mana yang relevan untuk pertanyaan pengetahuan.
    Return (needs_nist, needs_mitre). Bila sinyal hanya satu domain → domain itu saja;
    bila ambigu/lintas-domain/tak ada sinyal → keduanya (default aman)."""
    is_mitre = bool(_MITRE_DOMAIN_RE.search(query or ""))
    is_nist  = bool(_NIST_DOMAIN_RE.search(query or ""))
    if is_mitre and not is_nist:
        return False, True     # MITRE saja
    if is_nist and not is_mitre:
        return True, False     # NIST saja
    return True, True          # ambigu → ambil keduanya

def classify_intent(query: str) -> dict:
    """Klasifikasi intent + tentukan koleksi Qdrant yang perlu di-query.
    SATU sumber kebenaran — dipakai handler WebSocket (live) DAN skrip evaluasi
    (6a/6b) agar RAGAS benar-benar mengukur perilaku app.py."""
    technical    = is_technical_query(query)
    knowledge    = is_knowledge_query(query)
    log_analysis = is_log_analysis_query(query)

    if knowledge and not log_analysis:
        # Pertanyaan konsep murni → tidak perlu log; pilih domain KB yang relevan
        needs_logs = False
        needs_nist, needs_mitre = detect_kb_domain(query)
    else:
        needs_logs  = technical and not (knowledge and not log_analysis)
        needs_nist  = technical or knowledge
        needs_mitre = technical or knowledge

    return {
        "technical":   technical,
        "knowledge":   knowledge,
        "log_analysis": log_analysis,
        "needs_logs":  needs_logs,
        "needs_nist":  needs_nist,
        "needs_mitre": needs_mitre,
    }

# ============================================================
# CLEAN OUTPUT
# ============================================================
_MD_BOLD       = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC     = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_MD_HEADING    = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_CODE       = re.compile(r"`([^`]+)`")
_MD_THINK      = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MD_THINK_TAG  = re.compile(r"</?think>", re.IGNORECASE)
_MULTI_NL      = re.compile(r"\n{3,}")

def clean_output(text: str) -> str:
    """Bersihkan jawaban mentah LLM: buang blok <think>, strip markdown (**, #, `),
    rapikan baris kosong berlebih. Dipanggil sebelum guardrails_output()."""
    # Hapus blok think yang tertutup sempurna
    text = _MD_THINK.sub("", text)
    # Jika masih ada tag <think> atau </think> yang tersisa, strip tag-nya saja
    # (isi dipertahankan — lebih baik tampilkan reasoning daripada respons kosong)
    text = _MD_THINK_TAG.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()

# ============================================================
# FORMAT KONTEKS UNTUK EVALUASI RAGAS
# ============================================================
# Serialisasi hasil retrieve_context() menjadi list string yang dinilai RAGAS.
# Ada di app.py agar skrip evaluasi (6a) memanggil implementasi yang sama —
# bukan menduplikasi logika. Baris [Dataset] hanya relevan untuk pertanyaan
# tentang total/jumlah event atau rentang waktu, jadi di-gate dengan regex ini.
_DATASET_SCOPE_RE = re.compile(
    r"\b(total|jumlah|berapa\s+(banyak\s+)?(event|log|insiden|koneksi)|"
    r"rentang\s+waktu|periode|sejak\s+kapan)\b",
    re.IGNORECASE,
)

def format_contexts_for_eval(ctx: dict, query: str = "") -> list:
    """Format context hits ke list string untuk RAGAS.
    Menyertakan full_log (dstip, action) untuk akurasi evaluasi. Baris metadata
    [Dataset] hanya disertakan untuk pertanyaan scope dataset (total/rentang)."""
    parts = []
    total_db = app_state.get("logs_indexed", 0)
    md       = app_state.get("logs_metadata", {})
    if total_db > 0 and _DATASET_SCOPE_RE.search(query or ""):
        parts.append(
            f"[Dataset] Total log terindex: {total_db} event | "
            f"Rentang: {md.get('earliest','-')} s/d {md.get('latest','-')}"
        )
    for h in ctx.get("logs", []):
        p = h.payload or {}
        line = (
            f"[Log] {p.get('date','-')} {p.get('hour','-')} | "
            f"Rule {p.get('rule_id','-')} L{p.get('rule_level','-')} | "
            f"{p.get('rule_desc','-')} | Agent: {p.get('agent_name','-')} ({p.get('agent_ip','-')}) | "
            f"SrcIP: {p.get('srcip','-')}:{p.get('srcport','-')} | MITRE: {p.get('mitre_id','-')}"
        )
        flow_line = _format_flow_stats(p.get("flow_stats") or {})
        if flow_line:
            line += f" | Flow: {flow_line}"
        full_log = (p.get("full_log") or "").strip()
        if full_log:
            line += f" | RawLog: {full_log[:400]}"
        parts.append(line)
    for h in ctx.get("nist", []) + ctx.get("mitre", []):
        text = (h.payload or {}).get("text", "")
        if text.strip():
            parts.append(text)
    return [p for p in parts if p.strip()] or ["(tidak ada konteks)"]

# ============================================================
# PROMPT ADAPTIF DAN GENERASI RESPONS
# ============================================================
# ------------------------------------------------------------------
# Penyesuaian bentuk jawaban (adaptif, tidak kaku)
# ------------------------------------------------------------------
# Default jawaban = paragraf mengalir. Namun kalau analis meminta bentuk POIN/daftar
# secara eksplisit, sistem MENDETEKSI-nya di sisi Python (deterministik) lalu menyuntik
# instruksi tegas tanpa syarat di paling akhir prompt user. Ini jauh lebih andal
# ketimbang menyerahkan penalaran "kalau diminta poin maka..." ke LLM kecil (Qwen3-8B)
# di tengah prompt panjang yang penuh larangan. Gaya lain (sederhana/ringkas/rinci)
# ditangani lewat instruksi adaptif di SYSTEM_MSG & guidance, bukan lewat suntikan ini.
_POINTFORM_RE = re.compile(
    r"(poin[\s-]*poin|dalam\s+poin|bentuk\s+poin|per\s+poin|jadi\s+poin|"
    r"buatkan?\s+poin|pakai\s+poin|dengan\s+poin|secara\s+poin|"
    r"bullet|butir|rincikan|diperinci|perinci|"
    r"\bdaftar\b|listkan|\blist\b|bernomor|penomoran|numbered|"
    r"langkah[\s-]langkah|langkah\s+demi\s+langkah)",
    re.IGNORECASE,
)

def _wants_pointform(query: str) -> bool:
    """True kalau analis eksplisit meminta jawaban dalam bentuk poin/daftar."""
    return bool(_POINTFORM_RE.search(query or ""))

# Instruksi tegas, ditempel di PALING AKHIR prompt user hanya saat _wants_pointform().
_POINTFORM_DIRECTIVE = (
    "\n\nPERMINTAAN BENTUK JAWABAN (WAJIB DIIKUTI): analis meminta jawaban dalam "
    "bentuk poin. Susun SELURUH jawaban sebagai daftar berpoin memakai tanda hubung "
    "(- ) atau penomoran (1., 2., 3.), satu poin tiap baris. Khusus untuk jawaban ini, "
    "abaikan instruksi 'paragraf mengalir' dan semua larangan penomoran/daftar/struktur "
    "di atas. JANGAN menjawab dalam bentuk paragraf. Mulai langsung dari poin pertama."
)

def _style_directive(query: str) -> str:
    """Directive dinamis untuk ditempel di akhir prompt user. Default (paragraf) => ''."""
    return _POINTFORM_DIRECTIVE if _wants_pointform(query) else ""

SYSTEM_MSG = (
    "Kamu adalah SOCA (Security Operations Center Assistant), asisten keamanan siber "
    "untuk tim SOC PUSDATIN Kementerian Pertanian. Kamu ramah, profesional, dan "
    "SELALU berbicara dalam Bahasa Indonesia.\n\n"
    "ATURAN MUTLAK:\n"
    "1. Jawab HANYA berdasarkan data yang diberikan di prompt user (DATA LOG WAZUH, "
    "REFERENSI NIST CSF, REFERENSI MITRE ATT&CK). Ini adalah satu-satunya sumber "
    "kebenaran.\n"
    "2. JANGAN PERNAH menggunakan pengetahuan internal kamu tentang teknik MITRE, "
    "NIST CSF, atau topik keamanan apa pun. Pengetahuan kamu MUNGKIN SALAH atau "
    "KADALUARSA. SELALU pakai data yang diberikan.\n"
    "3. Kalau ditanya tentang ID teknik MITRE (misal T1574.009), BACA payload "
    "REFERENSI MITRE ATT&CK di prompt. Ambil nama teknik, taktik, deskripsi, dan "
    "mitigasi PERSIS dari sana. Jangan ubah atau tambah dari memori kamu.\n"
    "4. Kalau data yang diberikan tidak cocok dengan ID yang ditanya (misal user "
    "tanya T1574.009 tapi referensi MITRE berisi T1234.567), katakan terus terang "
    "bahwa data spesifik tidak ditemukan di knowledge base.\n"
    "5. Jangan gunakan simbol markdown (**, *, #, backtick). Secara DEFAULT, tulis "
    "jawaban dalam bentuk paragraf yang mengalir. Namun sesuaikan bentuk dan gaya "
    "jawaban dengan yang diminta analis: kalau ia minta poin atau daftar, jawab dalam "
    "poin memakai tanda hubung (- ) atau penomoran (1., 2.); kalau ia minta penjelasan "
    "yang sederhana, ringkas, atau lebih rinci, ikuti gaya itu. Tanpa permintaan bentuk "
    "khusus, tetap gunakan paragraf mengalir.\n"
    "6. ATURAN BAHASA: Gunakan Bahasa Indonesia untuk kalimat penjelasan dan analisismu. "
    "Namun JANGAN terjemahkan istilah teknis — biarkan tetap dalam bahasa Inggris apa adanya. "
    "Yang TIDAK boleh diterjemahkan: nama teknik MITRE (contoh: 'Unsecured Credentials', "
    "'Data Encoding', 'Brute Force'), nama taktik (contoh: 'credential-access', "
    "'command-and-control', 'execution'), nama mitigasi (contoh: 'Encrypt Sensitive "
    "Information', 'Password Policies'), nama fungsi/kategori NIST (contoh: 'DETECT', "
    "'Continuous Monitoring'), nama platform, nama tool, nama protokol, dan kode ID "
    "(T1059, DE.CM-01, dsb.). "
    "Yang WAJIB dalam Bahasa Indonesia: kalimat penjelasanmu sendiri, kalimat deskripsi "
    "panjang dari referensi (terjemahkan ke Indonesia), dan seluruh struktur jawabanmu."
)

def build_casual_prompt(query: str, history: list) -> str:
    """Prompt paling ringan — untuk sapaan/obrolan umum, tanpa retrieval Qdrant."""
    history_text = ""
    for role, msg in history[-2:]:
        history_text += f"{role}: {msg}\n"
    return (
        f"Percakapan sebelumnya:\n{history_text if history_text else '(kosong)'}\n\n"
        f"Pengguna bertanya: {query}\n\n"
        f"Jawab secara natural dan ringkas. Kalau sapaan, balas sapaan. "
        f"Kalau pertanyaan umum keamanan siber, jelaskan singkat."
        + _style_directive(query)
    )

def build_knowledge_prompt(query: str, context: dict, history: list) -> str:
    """
    Prompt untuk pertanyaan pengetahuan/faktual — jawab langsung dan ringkas.
    Tidak pakai template incident analysis (ringkasan/timeline/pola/urgensi).
    """
    history_text = ""
    for role, msg in history[-2:]:
        history_text += f"{role}: {msg}\n"

    nist_section  = "".join(_format_nist_hit(i, h) for i, h in enumerate(context["nist"], 1))
    mitre_section = "".join(_format_mitre_hit(i, h) for i, h in enumerate(context["mitre"], 1))

    if not nist_section:
        nist_section  = "(Tidak ada referensi NIST CSF yang cocok.)"
    if not mitre_section:
        mitre_section = "(Tidak ada referensi MITRE ATT&CK yang cocok.)"

    mitre_ids = _extract_mitre_ids(query)
    nist_ids  = _extract_nist_ids(query)
    ids_mentioned = mitre_ids + nist_ids

    # Deteksi apakah ini pertanyaan mitigasi/rekomendasi
    _MITIGATION_RE = re.compile(
        r"\b(mengatasi|mencegah|menangani|melindungi|memitigasi|merespons|menghadapi|"
        r"rekomendasi|mitigasi|solusi|penanganan|pencegahan|langkah|cara(nya)?)\b",
        re.IGNORECASE
    )
    is_mitigation = bool(_MITIGATION_RE.search(query))

    if ids_mentioned:
        id_str = ", ".join(ids_mentioned)
        instruction = (
            f"Cari entri dengan ID {id_str} di referensi di atas.\n"
            f"- Kalau ADA: jawab langsung — sebutkan nama, taktik, deskripsi singkat, dan mitigasi. "
            f"Ambil PERSIS dari field payload, jangan mengarang.\n"
            f"- Kalau TIDAK ADA: katakan terus terang tidak ditemukan di knowledge base."
        )
    elif is_mitigation:
        instruction = (
            "Pertanyaan ini meminta REKOMENDASI atau LANGKAH MITIGASI.\n"
            "Secara DEFAULT, tulis jawaban dalam paragraf yang mengalir yang menjelaskan "
            "jenis ancaman sekaligus cara pencegahan/mitigasinya secara ringkas, dengan "
            "ID referensi (contoh: DE.CM-01, M1050) disebut natural dalam kalimat. Kalau "
            "analis meminta poin atau langkah-langkah, sajikan sebagai daftar (tanda "
            "hubung - atau penomoran 1., 2.).\n"
            "Tetap berlaku:\n"
            "- JANGAN buat timeline atau analisis log — ini bukan pertanyaan insiden.\n"
            "- JANGAN sebut 'tidak ada log yang ditemukan'.\n"
            "- JANGAN gunakan template analisis kejadian."
        )
    else:
        instruction = (
            "Jawab pertanyaan ini secara langsung dan fokus pada apa yang ditanya.\n"
            "Manfaatkan SEMUA referensi di atas yang relevan sebagai dasar jawaban. "
            "Jika referensi tidak memuat aspek yang persis ditanya, tetap jawab dengan "
            "informasi terdekat yang tersedia di referensi — jangan menolak menjawab.\n"
            "JANGAN menambahkan fakta yang tidak ada di referensi. JANGAN menulis komentar "
            "tentang apa yang 'ditekankan' atau 'tidak dibahas' oleh referensi — cukup "
            "sampaikan jawabannya secara langsung.\n"
            "Tidak perlu template ringkasan/timeline/pola serangan — cukup jawab apa yang ditanya."
        )

    return f"""Riwayat percakapan:
{history_text if history_text else "(kosong)"}

=== REFERENSI NIST CSF 2.0 (Top-{TOP_K_NIST}) ===
{nist_section}

=== REFERENSI MITRE ATT&CK (Top-{TOP_K_MITRE}) ===
{mitre_section}

=== PERTANYAAN ===
{query}

{instruction}

Aturan:
- Jawab langsung, ringkas, dan faktual.
- Dasarkan HANYA pada referensi di atas. JANGAN mengarang.
- Jangan gunakan simbol markdown (**, ##, backtick).
- Kalimat penjelasan dan deskripsi panjang: tulis dalam Bahasa Indonesia. Nama teknik, nama taktik, nama mitigasi, nama fungsi NIST, dan istilah teknis lainnya: biarkan dalam bahasa Inggris apa adanya.{_style_directive(query)}
"""

def _format_log_hit(i: int, hit) -> str:
    """Format detail 1 log (versi panjang, per-baris berlabel). Saat ini tidak
    dipanggil di pipeline manapun — build_technical_prompt() pakai versi
    _format_log_hit_compact() di bawah. Disimpan sebagai alternatif format."""
    p = hit.payload or {}
    # Label berbasis waktu + deskripsi rule — bukan "Log 1/2/3"
    # agar LLM merujuk kejadian secara kontekstual, bukan nomor urut internal
    ts    = f"{p.get('date','-')} {p.get('hour','-')}"
    desc  = p.get('rule_desc', '-')
    label = f"[Kejadian: {ts} | {desc}]"
    return (
        f"\n{label}\n"
        f"  Agent     : {p.get('agent_name','-')} (IP {p.get('agent_ip','-')})\n"
        f"  Rule      : {p.get('rule_id','-')} level={p.get('rule_level','-')} "
        f"fired={p.get('rule_fired','-')}x\n"
        f"  Groups    : {p.get('rule_groups','-')}\n"
        f"  MITRE     : {p.get('mitre_id','-')} | Tactic: {p.get('mitre_tactic','-')} "
        f"| Technique: {p.get('mitre_technique','-')}\n"
        f"  Source    : {p.get('srcip','-')}:{p.get('srcport','-')} "
        f"user={p.get('srcuser','-')} -> dst user={p.get('dstuser','-')}\n"
        f"  Protocol  : {p.get('protocol','-')}  URL: {p.get('url','-')}\n"
        f"  Command   : {p.get('command','-')}\n"
        f"  Src SIEM  : {p.get('source', 'wazuh_log')}\n"
    )

def _format_log_hit_compact(hit) -> str:
    """Format ringkas 1 log — digunakan saat memuat banyak log sekaligus ke prompt."""
    p = hit.payload or {}
    line = (
        f"[{p.get('date','-')} {p.get('hour','-')}]"
        f" L{p.get('rule_level','-')}"
        f" Rule:{p.get('rule_id','-')}"
        f" {p.get('rule_desc','-')[:65]}"
        f" | Agent:{p.get('agent_name','-')}({p.get('agent_ip','-')})"
        f" | Src:{p.get('srcip','-')}:{p.get('srcport','-')}"
        f" | MITRE:{p.get('mitre_id','-')}"
    )
    # Wakil dari sekelompok log kembar — sebutkan jumlahnya supaya fakta "terjadi
    # berulang" tetap sampai ke LLM meski barisnya tidak diulang.
    dup = p.get("_dup_count") or 1
    if dup > 1:
        line += f" | Berulang:{dup}x kejadian serupa"
    if p.get('command'):
        line += f" | CMD:{p.get('command','')[:60]}"
    if p.get('url'):
        line += f" | URL:{p.get('url','')[:60]}"
    # flow_stats (dst/bytes/connections/duration) diekstrak saat ingest, sebelum
    # full_log dipotong — pastikan fakta kunci ini selalu tampil walau RawLog panjang.
    flow_line = _format_flow_stats(p.get("flow_stats") or {})
    if flow_line:
        line += f" | Flow:{flow_line}"
    # full_log mengandung dstip, action (Drop/Accept), dstport untuk log QRadar/CEF
    full_log = (p.get("full_log") or "").strip()
    if full_log:
        line += f" | RawLog:{full_log[:400]}"
    return line + "\n"


# ------------------------------------------------------------------
# PEMBERSIH & PEMOTONG TEKS KNOWLEDGE BASE
# ------------------------------------------------------------------
# Payload `text` di Qdrant kaya dan terstruktur, tapi versi sebelumnya hanya
# memotong 800 karakter pertama secara buta. Audit koleksi menunjukkan:
#   - 206 char pertama selalu berisi header (Technique/Tactics/Platforms/
#     Reference) yang SUDAH dicetak sebagai field terpisah di bawah -> mubazir.
#   - Bagian "Mitigations:" hanya masuk pada 36 dari 691 teknik (5%), padahal di
#     situlah saran konkret berada (baris "| Context: ..." tiap mitigasi).
#   - Ada 1.359 penanda "(Citation: ...)" dan 635 link markdown yang ikut memakan
#     jatah karakter tanpa memberi informasi.
# Untuk NIST, jatah 700 char habis oleh Function/Category (juga sudah dicetak
# terpisah) lalu tersisa untuk "Informative References" -- daftar pemetaan
# lintas-framework (CCMv4.0, CIS Controls, CRI Profile) yang tidak berguna untuk
# menjawab analis, sementara "Recommended Actions" justru terpotong.
#
# Helper di bawah memperbaiki hal itu: ambil per-SEKSI, bersihkan derau, dan
# alokasikan jatah karakter ke bagian yang benar-benar menjawab pertanyaan.
_CITATION_RE = re.compile(r"\s*\(Citation:[^)]*\)")
_MDLINK_RE   = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]*\)")
_WS_RE       = re.compile(r"[ \t]{2,}")

def _clean_kb_text(text: str) -> str:
    """Buang derau kutipan & link markdown supaya jatah karakter prompt terpakai
    untuk isi, bukan referensi. '[Valid Accounts](url)' -> 'Valid Accounts'."""
    t = _CITATION_RE.sub("", text or "")
    t = _MDLINK_RE.sub(r"\1", t)
    t = _WS_RE.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _kb_section(text: str, name: str, stops: tuple) -> str:
    """Ambil isi satu seksi (mis. 'Description:') sampai bertemu salah satu
    penanda seksi berikutnya. Return '' kalau seksi tidak ada."""
    t = text or ""
    i = t.find(name)
    if i < 0:
        return ""
    i += len(name)
    end = len(t)
    for s in stops:
        j = t.find(s, i)
        if 0 <= j < end:
            end = j
    return t[i:end].strip()

def _shorten(text: str, limit: int) -> str:
    """Potong di batas kalimat/baris terdekat agar tidak terputus di tengah kata."""
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    for sep in (". ", "\n", " "):
        k = cut.rfind(sep)
        if k > limit * 0.6:
            return cut[:k + (1 if sep == ". " else 0)].strip() + " ..."
    return cut.strip() + " ..."

_MITRE_STOPS = ("\nMitigations:", "\nDetection Analytics:")
_NIST_STOPS  = ("\nInformative References:", "\nRecommended Actions:")

# Jatah karakter per bagian. Total mirip anggaran lama (800/700) tetapi diisi
# konten yang menjawab, bukan header duplikat dan daftar referensi.
MITRE_DESC_CHARS = 620
MITRE_MITI_CHARS = 520
NIST_GUIDE_CHARS = 320
NIST_ACTION_CHARS = 520

def _mitre_mitigation_advice(text: str, limit: int) -> str:
    """Ringkas seksi 'Mitigations:' jadi 'Nama: saran konkret'.

    Tiap mitigasi di payload berbentuk:
        - <Nama>: <penjelasan panjang & generik> | Context: <saran spesifik>
    Bagian setelah '| Context:' itulah yang paling actionable bagi analis SOC
    (mis. 'Set account lockout policies after N failed login attempts'), jadi
    bagian itu yang diprioritaskan; penjelasan generiknya dibuang."""
    sec = _kb_section(text, "\nMitigations:", ("\nDetection Analytics:",))
    if not sec:
        return ""
    out, used = [], 0
    for line in sec.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        line = line[2:]
        nama, _, sisa = line.partition(":")
        ctx = ""
        if "| Context:" in sisa:
            ctx = sisa.split("| Context:", 1)[1].strip()
        else:
            ctx = sisa.strip()
        if not ctx:
            continue
        item = f"{nama.strip()}: {_shorten(ctx, 190)}"
        if used + len(item) > limit:
            # Kalau item PERTAMA saja sudah melebihi jatah, tetap sertakan versi
            # pendeknya — lebih baik satu saran terpotong daripada tidak ada sama
            # sekali (kasus ini menimpa 68 teknik pada uji koleksi).
            if not out:
                out.append(_shorten(item, limit))
            break
        out.append(item)
        used += len(item)
    return "; ".join(out)

def _format_mitre_hit(i: int, hit) -> str:
    """Format 1 hasil retrieval MITRE ATT&CK jadi blok teks untuk prompt LLM.
    Field terstruktur dicetak sekali; isi naratif diambil per-seksi dan sudah
    dibersihkan, sehingga saran mitigasi ikut sampai ke LLM (dulu hampir selalu
    terpotong)."""
    p = hit.payload or {}
    tech_id     = p.get("technique_id") or p.get("id") or "-"
    name        = p.get("name", "")
    tactics     = _join(p.get("tactics", []))
    platforms   = _join(p.get("platforms", []))
    parent      = p.get("parent_technique", "")
    is_sub      = p.get("is_subtechnique", False)
    mitigations = _join(p.get("mitigation_names", []))
    log_sources = _join(p.get("log_sources", []))
    url         = p.get("url", "")

    raw   = _clean_kb_text(p.get("text") or "")
    desc  = _shorten(_kb_section(raw, "Description:", _MITRE_STOPS), MITRE_DESC_CHARS)
    advice = _mitre_mitigation_advice(raw, MITRE_MITI_CHARS)

    out = (f"\n[MITRE ATT&CK {i} | score={hit.score:.3f}]\n"
           f"  Technique ID  : {tech_id}\n"
           f"  Name          : {name}\n"
           f"  Tactics       : {tactics}\n"
           f"  Platforms     : {platforms}\n")
    if is_sub and parent:
        out += f"  Sub-technique : YES (parent: {parent})\n"
    if mitigations:
        out += f"  Mitigations   : {mitigations}\n"
    if log_sources:
        out += f"  Log Sources   : {log_sources}\n"
    if url:
        out += f"  Reference     : {url}\n"
    if desc:
        out += f"  Description   : {desc}\n"
    if advice:
        out += f"  Mitigation Detail : {advice}\n"
    return out

def _format_nist_hit(i: int, hit) -> str:
    """Format 1 hasil retrieval NIST CSF jadi blok teks untuk prompt LLM.
    'Informative References' (pemetaan lintas-framework) sengaja DIBUANG karena
    tidak menjawab pertanyaan analis, dan jatahnya dialihkan ke 'Recommended
    Actions' yang berisi langkah konkret."""
    p = hit.payload or {}
    sub_id   = p.get("sub_id") or p.get("id") or "-"
    function = p.get("function", "")
    category = p.get("category", "")
    name     = p.get("name", "")

    raw   = _clean_kb_text(p.get("text") or "")
    guide = _shorten(_kb_section(raw, "Guideline:", _NIST_STOPS), NIST_GUIDE_CHARS)
    acts  = _shorten(_kb_section(raw, "Recommended Actions:",
                                 ("\nInformative References:",)), NIST_ACTION_CHARS)

    out = (f"\n[NIST CSF {i} | score={hit.score:.3f}]\n"
           f"  Subcategory ID : {sub_id}\n"
           f"  Function       : {function}\n")
    if category:
        out += f"  Category       : {category}\n"
    if name:
        out += f"  Name           : {name}\n"
    if guide:
        out += f"  Guideline      : {guide}\n"
    if acts:
        out += f"  Recommended Actions : {acts}\n"
    if not guide and not acts:   # jaga-jaga kalau format teks berbeda
        out += f"  Description    : {_shorten(raw, 700)}\n"
    return out

def build_technical_prompt(query: str, context: dict, history: list,
                           is_knowledge: bool = False) -> str:
    """
    Prompt utama untuk analisis log/teknis — menyusun 3 bagian konteks (log, NIST,
    MITRE) + instruksi (`guidance`) yang berbeda tergantung jenis pertanyaan:
    - is_knowledge + ID spesifik disebut  -> instruksi cari ID PERSIS di referensi
    - is_knowledge tanpa ID               -> instruksi jawab konseptual dari KB
    - bukan knowledge (log_analysis)      -> instruksi analisis log, larang beri
      rekomendasi NIST/MITRE kecuali diminta
    `fallback` menambah instruksi darurat jika retrieval kosong (log/KB tidak ada).
    """
    history_text = ""
    for role, msg in history[-2:]:
        history_text += f"{role}: {msg}\n"

    log_section   = "".join(_format_log_hit_compact(h) for h in context["logs"])
    nist_section  = "".join(_format_nist_hit(i, h) for i, h in enumerate(context["nist"], 1))
    mitre_section = "".join(_format_mitre_hit(i, h) for i, h in enumerate(context["mitre"], 1))

    has_logs  = bool(log_section)
    has_nist  = bool(nist_section)
    has_mitre = bool(mitre_section)
    has_any   = has_logs or has_nist or has_mitre

    if not log_section:
        log_section   = "(Tidak ada log relevan yang ditemukan untuk query ini.)"
    if not nist_section:
        nist_section  = "(Tidak ada referensi NIST CSF yang cocok.)"
    if not mitre_section:
        mitre_section = "(Tidak ada referensi MITRE ATT&CK yang cocok.)"

    if is_knowledge:
        mitre_ids_in_q = _extract_mitre_ids(query)
        nist_ids_in_q  = _extract_nist_ids(query)
        ids_mentioned  = mitre_ids_in_q + nist_ids_in_q

        if ids_mentioned:
            id_str = ", ".join(ids_mentioned)
            guidance = f"""Jenis pertanyaan ini adalah PERTANYAAN PENGETAHUAN dengan ID SPESIFIK: {id_str}.

LANGKAH WAJIB sebelum menjawab:
1. Cari di REFERENSI MITRE ATT&CK / NIST CSF di atas — apakah ada entri dengan ID PERSIS {id_str}?
2. Kalau ADA: jawab dengan menyalin field dari entri tersebut (name, tactics, description, mitigations).
   - Sebut nama teknik PERSIS dari field "name" di payload.
   - Sebut taktik PERSIS dari field "tactics" di payload.
   - Sebut mitigasi PERSIS dari field "mitigation_names" atau dari deskripsi.
   - JANGAN tambah informasi dari pengetahuan kamu sendiri.
3. Kalau TIDAK ADA entri yang ID-nya cocok: katakan terus terang "Data untuk {id_str} tidak ditemukan di knowledge base. Yang paling mirip dari hasil pencarian adalah: [sebut entri terdekat dengan score tertinggi]". JANGAN mengarang jawaban.

Format jawaban:
- Mulai dengan: "[ID] adalah [name dari payload]."
- Lanjutkan dengan deskripsi singkat dari payload.
- Sebut taktik dan platform.
- Tutup dengan rekomendasi mitigasi dari payload."""
        else:
            guidance = """Jenis pertanyaan ini adalah PERTANYAAN PENGETAHUAN/KONSEPTUAL.

Cara menjawab:
1. Gunakan REFERENSI MITRE ATT&CK dan/atau NIST CSF 2.0 di atas sebagai sumber utama.
2. Jelaskan konsep dengan bahasa Indonesia yang jelas, tapi dasarkan PADA ISI REFERENSI di atas.
3. Data log TIDAK diperlukan untuk pertanyaan ini.
4. JANGAN tambah informasi yang tidak ada di referensi."""
    else:
        guidance = """Jenis pertanyaan ini adalah ANALISIS LOG.

CARA MENJAWAB:
- Jawab LANGSUNG apa yang ditanya. Mulai kalimat pertama dengan fakta utama.
- Gunakan angka spesifik, nama agen, IP, port, rule_id, rule_desc, dan nilai dari log di atas.
- JANGAN gunakan "Log 1", "Log 2" — rujuk dengan waktu dan deskripsi kejadian.
- JANGAN tambahkan rekomendasi NIST/MITRE atau langkah mitigasi kecuali ditanya.
- DEFAULT jawab dalam paragraf yang mengalir tanpa header atau penomoran. Kalau analis meminta bentuk lain (poin, langkah-langkah, ringkas, atau sederhana), ikuti permintaan itu.
- Untuk pertanyaan jumlah/statistik: gunakan info DATASET di atas untuk angka total keseluruhan, dan gunakan log yang tersedia untuk detail distribusi atau contoh.
- Untuk laporan lengkap terstruktur, pengguna bisa pakai /full_analyze."""

    if not has_any:
        fallback = ('\nKarena tidak ada data log maupun referensi NIST/MITRE yang cocok, '
                    'katakan terus terang bahwa informasi tidak tersedia. '
                    'Sarankan analis memperluas rentang hari dengan /set days '
                    'atau memeriksa apakah knowledge base sudah ter-index.')
    elif is_knowledge and (has_nist or has_mitre):
        fallback = ('\nFokus pada referensi NIST/MITRE di atas untuk menjawab. '
                    'Log kosong itu wajar untuk pertanyaan konseptual — jangan bilang '
                    '"data tidak ditemukan".')
    elif not is_knowledge and not has_logs:
        fallback = ('\nTidak ada log relevan. Katakan "tidak ditemukan log yang cocok" '
                    'dan sarankan /set days untuk memperluas rentang. '
                    'Kalau referensi NIST/MITRE relevan, tetap berikan konteks umum.')
    else:
        fallback = ''

    n_logs   = len(context["logs"])
    total_db = app_state["logs_indexed"]
    md_meta  = app_state["logs_metadata"]
    earliest = md_meta.get("earliest", "-")
    latest   = md_meta.get("latest", "-")

    return f"""Riwayat percakapan:
{history_text if history_text else "(kosong)"}

=== INFO DATASET ===
Total log terindex di database: {total_db} event
Rentang waktu: {earliest} s/d {latest}

=== DATA LOG WAZUH ({n_logs} log relevan diambil) ===
{log_section}

=== REFERENSI NIST CSF 2.0 (Top-{TOP_K_NIST}) ===
{nist_section}

=== REFERENSI MITRE ATT&CK (Top-{TOP_K_MITRE}) ===
{mitre_section}

=== PERTANYAAN ANALIS ===
{query}

{guidance}

Aturan umum:
- Dasarkan jawaban HANYA pada data di atas. JANGAN mengarang.
- Jangan gunakan simbol markdown (**, ##, backtick).
- Kalimat penjelasan dan deskripsi panjang: tulis dalam Bahasa Indonesia. Nama teknik, nama taktik, nama mitigasi, nama fungsi NIST, dan istilah teknis lainnya: biarkan dalam bahasa Inggris apa adanya.{fallback}{_style_directive(query)}
"""

def build_full_analyze_prompt(query: str, context: dict, history: list) -> str:
    """
    Prompt untuk perintah /full_analyze — analisis komprehensif seluruh log
    menggunakan format laporan terstruktur dengan semua header wajib.
    """
    history_text = ""
    for role, msg in history[-2:]:
        history_text += f"{role}: {msg}\n"

    log_section   = "".join(_format_log_hit_compact(h) for h in context["logs"])
    nist_section  = "".join(_format_nist_hit(i, h) for i, h in enumerate(context["nist"], 1))
    mitre_section = "".join(_format_mitre_hit(i, h) for i, h in enumerate(context["mitre"], 1))

    if not log_section:
        log_section   = "(Tidak ada log di Qdrant. Jalankan /reload untuk mengambil log dari VPS.)"
    if not nist_section:
        nist_section  = "(Tidak ada referensi NIST CSF yang cocok.)"
    if not mitre_section:
        mitre_section = "(Tidak ada referensi MITRE ATT&CK yang cocok.)"

    n_logs = len(context["logs"])
    guidance = """FORMAT JAWABAN WAJIB — gunakan persis header berikut secara berurutan, tanpa ada yang dilewati:

Ringkasan Kejadian:
[Jelaskan apa yang terjadi, kapan, dan dari IP/user mana. Pakai tanggal & jam dari log.
Sebutkan jenis serangan atau aktivitas yang dominan secara ringkas.]

Timeline:
[Urutkan kejadian dari yang paling awal ke paling akhir berdasarkan waktu di log.
Satu baris per kejadian, format: <tanggal> <jam> — <deskripsi singkat kejadian>.]

Pola Serangan:
[Identifikasi IP, user, rule, atau teknik yang muncul berulang kali.
Jelaskan pola atau urutan yang terlihat dari data log di atas.]

Sumber Serangan & Aset Terdampak:
[Daftar semua IP sumber (penyerang atau pengirim traffic mencurigakan) dan IP/host tujuan
yang terdampak berdasarkan field srcip, dstip, agent_ip di log.
Format yang diharapkan:
  Sumber (Source):
  - <IP>:<port> — <keterangan singkat jika ada>
  Target / Aset Terdampak:
  - <IP atau hostname>:<port> — <keterangan singkat jika ada>
Catatan: informasi ini dapat digunakan langsung untuk blocking via Cloudflare atau firewall.
Jika tidak ada data IP yang jelas di log, tulis: "Data IP tidak tersedia di log yang diambil."]

Mapping MITRE ATT&CK:
[Sebut ID teknik & taktik dari field MITRE di log. Rujuk referensi MITRE di atas jika relevan.
Jika field MITRE di log kosong, tulis: "Tidak ada mapping MITRE ATT&CK yang tersedia di log ini."]

Kesimpulan Serangan:
[Nyatakan status akhir serangan secara eksplisit berdasarkan data log di atas.
Gunakan SALAH SATU label berikut:
  - BERHASIL     : ada indikasi akses berhasil, eksekusi berhasil, atau data keluar
  - GAGAL / DIBLOKIR : log menunjukkan koneksi ditolak, alert terpicu tanpa tindak lanjut, atau rule blocked
  - TIDAK DAPAT DITENTUKAN : data log tidak cukup untuk menyimpulkan hasil akhir
Sertakan satu kalimat ringkas mengenai dampak aktual yang terdeteksi dari log.]

Proses & Path Malware:
[Jika log mengandung data command, path file, nama proses, nama executable, atau
aktivitas filesystem yang mencurigakan — sebutkan secara spesifik di sini.
Contoh: path "/tmp/evil.sh", perintah "curl http://...", proses "powershell.exe -enc ...".
Jika tidak ada data proses atau path di log, tulis:
"Tidak ada data command, proses, atau path malware yang terdeteksi di log ini."]

Rekomendasi & To-Do List:
Tingkat Urgensi: [Low / Medium / High / Critical]
[Satu kalimat alasan penentuan level urgensi, mengacu pada rule_level dan dampak aktual.]

Tim Keamanan Siber:
[ ] <aksi konkret — contoh: Blokir IP X.X.X.X di firewall perimeter> — Urgensi: High
[ ] <aksi konkret — contoh: Update rule deteksi untuk pattern query ini> — Urgensi: Medium
[ ] <aksi konkret — contoh: Lakukan threat hunting di endpoint terdampak> — Urgensi: High

Pemilik Aset:
[ ] <aksi konkret — contoh: Review log akses server di rentang waktu kejadian> — Urgensi: Medium
[ ] <aksi konkret — contoh: Ganti kredensial akun yang terdampak jika ada> — Urgensi: High

[Tulis minimal 2 aksi per peran. Aksi harus konkret dan langsung bisa dieksekusi.
Gunakan referensi NIST CSF dan MITRE ATT&CK di atas sebagai dasar rekomendasi,
tapi sampaikan dalam bahasa operasional yang mudah dipahami tanpa menyebut ID teknis.]

ATURAN TINGKAT URGENSI — gunakan DUA acuan:
A) Wazuh Rule Level:
   - rule_level 0-6   → Low
   - rule_level 7-11  → Medium
   - rule_level 12-14 → High
   - rule_level 15-16 → Critical

B) Dampak aktual (NIST SP 800-61r3):
   - Low     : aktivitas gagal/informatif, tidak ada bukti keberhasilan
   - Medium  : satu sistem terdampak, belum ada konfirmasi data keluar
   - High    : akses tidak sah terkonfirmasi, kredensial bocor
   - Critical: sistem kritis down, ransomware aktif, data keluar massal

Gunakan rule_level sebagai titik awal, sesuaikan berdasarkan dampak aktual.
Jika log hanya menunjukkan aktivitas gagal, TURUNKAN ke Low meskipun rule_level menengah.
SEMUA header di atas WAJIB ada dalam jawaban, meskipun datanya terbatas. Jangan melewati satu pun header."""

    return f"""Riwayat percakapan:
{history_text if history_text else "(kosong)"}

=== DATA LOG WAZUH ({n_logs} log — seluruh collection) ===
{log_section}

=== REFERENSI NIST CSF 2.0 (Top-{TOP_K_NIST}) ===
{nist_section}

=== REFERENSI MITRE ATT&CK (Top-{TOP_K_MITRE}) ===
{mitre_section}

=== FOKUS ANALISIS ===
{query}

{guidance}

Aturan umum:
- Dasarkan jawaban HANYA pada data di atas. JANGAN mengarang.
- Jangan gunakan simbol markdown (**, ##, backtick).
- Kalimat penjelasan dan deskripsi panjang: tulis dalam Bahasa Indonesia. Nama teknik, nama taktik, nama mitigasi, nama fungsi NIST, dan istilah teknis lainnya: biarkan dalam bahasa Inggris apa adanya.
"""


# ============================================================
# GENERATE VIA LLM
# ============================================================
# Qwen3 soft-switch: matikan mode thinking agar token budget tidak habis untuk
# reasoning internal (penyebab jawaban terpotong) dan tidak ada kebocoran teks
# reasoning mentah. clean_output tetap menyaring sisa tag <think> sebagai pengaman.
_NO_THINK = " /no_think" if "qwen" in LLM_MODEL.lower() else ""

def generate_answer(prompt: str, is_technical: bool, max_tokens: int = 0) -> str:
    """
    Panggil LLM (Qwen3-8B via LM Studio, OpenAI-compatible API).
    temperature rendah (0.1) untuk technical query — jawaban lebih deterministik/faktual;
    lebih tinggi (0.4) untuk casual — sedikit lebih natural/variatif.
    "/no_think" ditambahkan ke system prompt agar Qwen3 tidak membuang token budget
    untuk reasoning internal (penyebab umum jawaban terpotong pada model ini).
    """
    temperature = 0.1 if is_technical else 0.4
    _max = max_tokens or (1200 if is_technical else 1000)
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MSG + _NO_THINK},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=_max,
    )
    raw = response.choices[0].message.content or ""
    logger.debug("LLM raw (%d chars): %s", len(raw), raw[:120].replace("\n", " "))
    return raw

# ============================================================
# BACA STATE DARI QDRANT (dijalankan saat startup)
# ============================================================
def read_state_from_qdrant() -> None:
    """Isi ulang app_state (jumlah log, rentang tanggal, last_sync_ts) dari data
    yang SUDAH ADA di Qdrant saat server start — supaya /stat & auto-sync langsung
    akurat tanpa perlu /reload dulu kalau Qdrant sudah terisi dari sesi sebelumnya."""
    try:
        if not qdrant.collection_exists(COL_LOGS):
            logger.info("Collection '%s' belum ada. Jalankan /reload untuk indexing pertama.", COL_LOGS)
            app_state["logs_indexed"]  = 0
            app_state["logs_metadata"] = {"total": 0, "earliest": "-", "latest": "-"}
            return

        info  = qdrant.get_collection(COL_LOGS)
        count = info.points_count or 0
        app_state["logs_indexed"] = count
        logger.info("Collection '%s' ditemukan: %d vectors. Siap digunakan.", COL_LOGS, count)

        if count > 0:
            sample, _ = qdrant.scroll(
                collection_name=COL_LOGS,
                limit=min(count, 5000),
                with_payload=["timestamp"],
                with_vectors=False,
            )
            timestamps = []
            for p in sample:
                ts = (p.payload or {}).get("timestamp", "")
                if ts:
                    try:
                        timestamps.append(datetime.fromisoformat(ts[:19]))
                    except ValueError:
                        pass
            if timestamps:
                latest_ts = max(timestamps)
                app_state["logs_metadata"] = {
                    "total":    count,
                    "earliest": min(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
                    "latest":   latest_ts.strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Set last_sync_ts dari data Qdrant yang sudah ada
                # sehingga auto-sync dan /sync bisa langsung berjalan tanpa /reload dulu
                app_state["last_sync_ts"] = latest_ts
                logger.info("Rentang log: %s s/d %s — last_sync_ts diset ke %s",
                            app_state["logs_metadata"]["earliest"],
                            app_state["logs_metadata"]["latest"],
                            latest_ts.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                app_state["logs_metadata"] = {"total": count, "earliest": "-", "latest": "-"}

    except Exception as e:
        logger.error("Gagal membaca state dari Qdrant: %s", e)
        app_state["logs_indexed"]  = 0
        app_state["logs_metadata"] = {"total": 0, "earliest": "-", "latest": "-"}


# ============================================================
# RELOAD CHAIN
# ============================================================
def reload_chain(past_days: int = 7) -> bool:
    """Rantai penuh perintah /reload: tarik log VPS (load_logs_from_vps) lalu
    indeks ke Qdrant (index_logs_to_qdrant). Return False kalau tidak ada log ditemukan."""
    app_state["days_range"] = past_days
    logger.info("Reload chain: mengambil log %d hari terakhir dari VPS...", past_days)

    logs = load_logs_from_vps(past_days)
    if not logs:
        logger.warning("Tidak ada log ditemukan dari VPS.")
        app_state["logs_indexed"] = 0
        return False

    index_logs_to_qdrant(logs)
    return True


# ============================================================
# AUTO SYNC BACKGROUND TASK
# ============================================================
async def auto_sync_task():
    """
    Background task — jalankan incremental sync setiap sync_interval detik.
    Hanya mengambil log baru (lebih baru dari last_sync_ts) via tail.
    Versi ini tidak meninggalkan jejak log jika tidak ada log baru.
    """
    logger.info("Auto-sync dimulai (interval=%ds).", app_state["sync_interval"])
    while app_state["auto_sync_on"]:
        await asyncio.sleep(app_state["sync_interval"])

        # Overlap — diam saja, tidak perlu log
        if app_state["sync_in_progress"]:
            continue

        since_ts = app_state.get("last_sync_ts")
        if since_ts is None:
            continue  # Tunggu /reload — tidak perlu log berulang

        app_state["sync_in_progress"] = True
        try:
            loop = asyncio.get_event_loop()
            new_logs = await loop.run_in_executor(None, load_logs_since, since_ts)
            if new_logs:
                # Ada log baru — barulah log muncul
                logger.info("Auto-sync: %d log baru ditemukan, mulai indexing...", len(new_logs))
                await loop.run_in_executor(None, index_logs_to_qdrant, new_logs)
                logger.info("Auto-sync selesai. %d log baru. Total di Qdrant: %d",
                            len(new_logs), app_state["logs_indexed"])
            # Tidak ada log baru — tidak ada output apapun ke terminal
        except Exception as e:
            logger.error("Auto-sync error: %s", e)
        finally:
            app_state["sync_in_progress"] = False


# ============================================================
# FASTAPI LIFESPAN
# ============================================================
def _suppress_winerror_10054(loop, context):
    """Redam noise WinError 10054 (client putus koneksi paksa) di log Windows —
    error ini normal terjadi saat browser/tab ditutup, bukan bug aplikasi."""
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
        return
    loop.default_exception_handler(context)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown FastAPI: baca state Qdrant, nyalakan auto-sync background,
    matikan auto-sync dengan bersih saat shutdown (cancel + await task)."""
    asyncio.get_event_loop().set_exception_handler(_suppress_winerror_10054)
    logger.info("Starting SOCA server...")
    read_state_from_qdrant()
    logger.info("Guardrails status: regex lokal (prompt injection) aktif")

    # Mulai background auto-sync
    sync_task = asyncio.create_task(auto_sync_task())

    yield

    # Matikan auto-sync saat shutdown
    app_state["auto_sync_on"] = False
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down SOCA server...")

app = FastAPI(title="SOCA - SOC Chatbot PUSDATIN", lifespan=lifespan)

_UI_DIR = Path(__file__).parent / "ui"
app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# WEBSOCKET
# ============================================================
# Satu koneksi WebSocket = satu sesi chat. Loop utama menangani 2 jenis pesan:
#   1. Command (diawali "/") — /help, /reload, /sync, /stat, /debug, /clear_logs,
#      /set days, /clear_chat, /full_analyze — masing-masing punya blok if terpisah.
#   2. Pesan bebas — masuk ke pipeline RAG penuh: guardrails_input -> classify_intent
#      -> retrieve_context -> build_*_prompt -> generate_answer -> clean_output ->
#      guardrails_output -> kirim ke client.
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Auth via pesan pertama — hindari eksposur kredensial di URL/log server
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        auth_msg = json.loads(raw)
        if auth_msg.get("type") != "auth" or not verify_credentials(
            auth_msg.get("u", ""), auth_msg.get("p", "")
        ):
            await websocket.close(code=4001, reason="Unauthorized")
            return
    except (asyncio.TimeoutError, json.JSONDecodeError):
        await websocket.close(code=4001, reason="Auth gagal atau timeout")
        return

    chat_history: list[tuple[str, str]] = []

    try:
        no_log_hint = (
            "\n\nCATATAN: Belum ada log terindex. Ketik /reload untuk mengambil log dari Wazuh VPS."
            if app_state["logs_indexed"] == 0 else ""
        )
        await websocket.send_json({
            "role": "bot",
            "message": (
                f"Halo! Saya SOCA, asisten analisis log keamanan PUSDATIN.\n"
                f"Model         : {LLM_MODEL}\n"
                f"Log terindex  : {app_state['logs_indexed']} events\n"
                f"Rentang waktu : {app_state['logs_metadata'].get('earliest','-')} "
                f"s/d {app_state['logs_metadata'].get('latest','-')}\n"
                f"Ketik /help untuk daftar perintah."
                + no_log_hint
            )
        })

        while True:
            data  = (await websocket.receive_text()).strip()
            if not data:
                continue

            lower = data.lower()

            # ---------- COMMANDS ----------
            if lower == "/help":
                await websocket.send_json({"role": "bot", "message":
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    " SOCA — Daftar Perintah\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "\n"
                    "🔍 ANALISIS LOG\n"
                    "  /full_analyze    Analisis komprehensif SELURUH log dengan\n"
                    "                   laporan terstruktur: Ringkasan, Timeline,\n"
                    "                   Pola Serangan, MITRE, Rekomendasi, dsb.\n"
                    "                   Opsional: /full_analyze brute force SSH\n"
                    "\n"
                    "📊 DATA LOG\n"
                    "  /reload          Ambil log dari Wazuh VPS sejak N hari\n"
                    "                   terakhir dan tambahkan ke database.\n"
                    "                   Log lama di Qdrant tidak dihapus.\n"
                    "\n"
                    "  /sync            Ambil log baru secara manual sekarang\n"
                    "                   (tanpa menunggu interval 30 detik).\n"
                    "                   Hanya mengambil log lebih baru dari\n"
                    "                   timestamp terakhir yang tersimpan.\n"
                    "\n"
                    "  /clear_logs      Hapus SELURUH log dari database Qdrant.\n"
                    "                   Gunakan /reload setelah ini untuk\n"
                    "                   memuat ulang dari awal.\n"
                    "\n"
                    "  /set days <n>    Atur rentang hari yang digunakan saat\n"
                    "                   /reload berikutnya dijalankan.\n"
                    "                   Contoh: /set days 14\n"
                    "\n"
                    "📈 STATUS & INFO\n"
                    "  /stat            Tampilkan jumlah log, rentang waktu,\n"
                    "                   status auto-sync, dan info Guardrails.\n"
                    "\n"
                    "💬 PERCAKAPAN\n"
                    "  /clear_chat       Hapus riwayat percakapan sesi ini.\n"
                    "                   Data log di Qdrant tidak terpengaruh.\n"
                    "\n"
                    "  /help            Tampilkan menu ini.\n"
                    "\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Auto-sync aktif setiap 30 detik secara otomatis.\n"
                    "Untuk analisis bebas, ketik pertanyaan langsung — contoh:\n"
                    "  'Apakah ada serangan brute force hari ini?'\n"
                    "  'IP mana yang paling sering muncul di log?'\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                })
                continue

            if lower == "/reload":
                await websocket.send_json({"role": "bot",
                    "message": (
                        f"Mengambil log {app_state['days_range']} hari terakhir dari Wazuh VPS...\n"
                        f"Log lama di Qdrant tetap dipertahankan, hanya log baru yang ditambahkan.\n"
                        f"(Proses ini membutuhkan waktu beberapa menit)"
                    )})
                try:
                    loop = asyncio.get_event_loop()
                    ok = await loop.run_in_executor(None, reload_chain, app_state["days_range"])
                except Exception as e:
                    logger.exception("Reload error")
                    await websocket.send_json({"role": "bot",
                        "message": f"Reload gagal: {e}"})
                    continue
                if ok:
                    chat_history = []
                    md = app_state["logs_metadata"]
                    await websocket.send_json({"role": "bot", "message":
                        f"Reload selesai.\n"
                        f"Total log terindex : {app_state['logs_indexed']}\n"
                        f"Rentang            : {md.get('earliest','-')} s/d {md.get('latest','-')}"
                    })
                else:
                    await websocket.send_json({"role": "bot",
                        "message": "Reload gagal - tidak ada log ditemukan di VPS."})
                continue

            if lower == "/sync":
                since_ts = app_state.get("last_sync_ts")
                if since_ts is None:
                    await websocket.send_json({"role": "bot",
                        "message": "Belum ada data awal. Jalankan /reload dulu sebelum /sync."})
                    continue
                if app_state["sync_in_progress"]:
                    await websocket.send_json({"role": "bot",
                        "message": "Sync sedang berjalan, harap tunggu sebentar."})
                    continue
                await websocket.send_json({"role": "bot",
                    "message": f"Trigger sync manual sejak {since_ts.strftime('%Y-%m-%d %H:%M:%S')}..."})
                app_state["sync_in_progress"] = True
                try:
                    loop = asyncio.get_event_loop()
                    new_logs = await loop.run_in_executor(None, load_logs_since, since_ts)
                    if new_logs:
                        await loop.run_in_executor(None, index_logs_to_qdrant, new_logs)
                        await websocket.send_json({"role": "bot",
                            "message": f"Sync selesai. {len(new_logs)} log baru ditambahkan.\n"
                                       f"Total di Qdrant: {app_state['logs_indexed']}"})
                    else:
                        await websocket.send_json({"role": "bot",
                            "message": "Tidak ada log baru sejak sync terakhir."})
                except Exception as e:
                    await websocket.send_json({"role": "bot", "message": f"Sync error: {e}"})
                finally:
                    app_state["sync_in_progress"] = False
                continue

            if lower.startswith("/set days "):
                try:
                    new_days = int(data.split()[-1])
                    if not 1 <= new_days <= 365:
                        raise ValueError("Di luar rentang 1-365")
                    app_state["days_range"] = new_days
                    await websocket.send_json({"role": "bot", "message":
                        f"Rentang log diset ke {new_days} hari. Jalankan /reload untuk menerapkan."})
                except (ValueError, IndexError):
                    await websocket.send_json({"role": "bot",
                        "message": "Format salah. Gunakan: /set days <angka 1-365>"})
                continue

            if lower == "/stat":
                def _count(name):
                    try:
                        return qdrant.get_collection(name).points_count if qdrant.collection_exists(name) else 0
                    except Exception:
                        return 0
                md = app_state["logs_metadata"]
                last_sync = app_state.get("last_sync_ts")
                last_sync_str = last_sync.strftime("%Y-%m-%d %H:%M:%S") if last_sync else "belum ada"
                sync_status = (
                    "sedang berjalan..." if app_state["sync_in_progress"]
                    else f"aktif (setiap {app_state['sync_interval']}d)" if app_state["auto_sync_on"]
                    else "nonaktif"
                )
                await websocket.send_json({"role": "bot", "message":
                    f"Statistik sistem:\n"
                    f"Wazuh logs      : {_count(COL_LOGS)} vectors\n"
                    f"NIST CSF 2.0    : {_count(COL_NIST)} vectors\n"
                    f"MITRE ATT&CK    : {_count(COL_MITRE)} vectors\n"
                    f"Rentang waktu   : {app_state['days_range']} hari\n"
                    f"Log pertama     : {md.get('earliest','-')}\n"
                    f"Log terakhir    : {md.get('latest','-')}\n"
                    f"Last sync       : {last_sync_str}\n"
                    f"Auto-sync       : {sync_status}\n"
                    f"Guardrails      : Guardrails AI — PromptInjectionDetector ({GUARDRAILS_MODEL}, {'aktif' if _guardrails_available else 'fallback regex'})"
                })
                continue

            if lower in ("/clear", "/clear_chat"):
                chat_history = []
                await websocket.send_json({"role": "bot",
                    "message": "Riwayat percakapan dihapus."})
                continue

            # Perintah TERSEMBUNYI (tidak muncul di UI maupun /help), seperti /debug.
            # Meng-toggle Guardrails saat runtime — untuk eksperimen & menghemat
            # panggilan API LLM judge. Validasi panjang input/output tetap jalan.
            if lower.startswith("/guardrails"):
                global _guardrails_enabled
                arg = lower[len("/guardrails"):].strip()
                if arg == "on":
                    _guardrails_enabled = True
                    logger.info("Guardrails DIAKTIFKAN via /guardrails on.")
                    await websocket.send_json({"role": "bot",
                        "message": "Guardrails AKTIF. Deteksi prompt injection dijalankan pada input dan output."})
                elif arg == "off":
                    _guardrails_enabled = False
                    logger.warning("Guardrails DINONAKTIFKAN via /guardrails off.")
                    await websocket.send_json({"role": "bot",
                        "message": "Guardrails NONAKTIF. Deteksi prompt injection dilewati "
                                   "(validasi panjang tetap aktif). Ketik /guardrails on untuk mengaktifkan kembali."})
                else:
                    _status = "AKTIF" if _guardrails_enabled else "NONAKTIF"
                    await websocket.send_json({"role": "bot",
                        "message": f"Status Guardrails saat ini: {_status}.\n"
                                   f"Format: /guardrails on  atau  /guardrails off"})
                continue

            if lower.startswith("/debug "):
                debug_query = data[7:].strip()
                if not debug_query:
                    await websocket.send_json({"role": "bot",
                        "message": "Format: /debug <query>"})
                    continue
                try:
                    ctx = retrieve_context(debug_query)
                    lines = [f"DEBUG retrieval untuk: '{debug_query}'\n"]
                    lines.append(f"LOG HITS ({len(ctx['logs'])}):")
                    for h in ctx["logs"]:
                        p = h.payload or {}
                        score = getattr(h, 'score', 1.0) or 1.0
                        lines.append(f"  score={score:.3f} | rule={p.get('rule_id','-')} "
                                     f"| {p.get('rule_desc','-')[:60]}")
                    lines.append(f"\nNIST HITS ({len(ctx['nist'])}):")
                    for h in ctx["nist"]:
                        p = h.payload or {}
                        lines.append(f"  score={h.score:.3f} | {p.get('sub_id','-')} "
                                     f"| {p.get('name','-')[:60]}")
                    lines.append(f"\nMITRE HITS ({len(ctx['mitre'])}):")
                    for h in ctx["mitre"]:
                        p = h.payload or {}
                        lines.append(f"  score={h.score:.3f} | {p.get('technique_id','-')} "
                                     f"| {p.get('name','-')[:60]}")
                    await websocket.send_json({"role": "bot", "message": "\n".join(lines)})
                except Exception as e:
                    await websocket.send_json({"role": "bot",
                        "message": f"Debug error: {e}"})
                continue

            if lower == "/clear_logs":
                await websocket.send_json({"role": "bot",
                    "message": "Menghapus semua log dari Qdrant... Harap tunggu."})
                try:
                    clear_logs_collection()
                    await websocket.send_json({"role": "bot",
                        "message": "Semua log berhasil dihapus dari Qdrant.\n"
                                   "Gunakan /reload untuk mengambil log baru dari VPS."})
                except Exception as e:
                    logger.exception("Clear logs error")
                    await websocket.send_json({"role": "bot",
                        "message": f"Gagal menghapus log: {e}"})
                continue

            if lower.startswith("/full_analyze"):
                fa_query = data[len("/full_analyze"):].strip() or "Analisis komprehensif seluruh log keamanan"
                await websocket.send_json({"role": "bot",
                    "message": "Menganalisis seluruh log secara komprehensif..."})
                loop = asyncio.get_event_loop()
                try:
                    fa_logs  = await loop.run_in_executor(None, _fetch_all_logs_scroll) if qdrant.collection_exists(COL_LOGS) else []
                    fa_vec   = (await loop.run_in_executor(None, lambda: embedder.encode(fa_query, normalize_embeddings=True))).tolist()
                    fa_nist  = await loop.run_in_executor(None, lambda: _hybrid_search(COL_NIST,  fa_vec, TOP_K_NIST))  if qdrant.collection_exists(COL_NIST)  else []
                    fa_mitre = await loop.run_in_executor(None, lambda: _hybrid_search(COL_MITRE, fa_vec, TOP_K_MITRE)) if qdrant.collection_exists(COL_MITRE) else []
                    fa_ctx   = {"logs": fa_logs, "nist": fa_nist, "mitre": fa_mitre}
                    fa_prompt = build_full_analyze_prompt(fa_query, fa_ctx, chat_history)
                except Exception as e:
                    logger.exception("full_analyze retrieval error")
                    await websocket.send_json({"role": "bot",
                        "message": f"Gagal mengambil data: {e}"})
                    continue
                await websocket.send_json({"role": "bot",
                    "message": "Menyusun laporan analisis lengkap..."})
                try:
                    raw_answer = await loop.run_in_executor(None, lambda: generate_answer(fa_prompt, is_technical=True, max_tokens=3000))
                except Exception as e:
                    logger.exception("LLM error pada full_analyze")
                    await websocket.send_json({"role": "bot", "message": f"LLM error: {e}"})
                    continue
                cleaned   = clean_output(raw_answer)
                valid_out, answer = guardrails_output(cleaned)
                if not valid_out:
                    await websocket.send_json({"role": "bot",
                        "message": f"Jawaban tidak dapat diproses: {answer}"})
                    continue
                # Riwayat percakapan sengaja TIDAK disimpan: tiap pertanyaan dijawab
                # independen agar topik lama tidak nyangkut ke pertanyaan berikutnya.
                await websocket.send_json({"role": "bot", "message": answer})
                continue

            # ---------- RAG PIPELINE ----------
            # Placeholder dikirim SEBELUM guardrails — pemeriksaan Lapis 2 (LLM
            # judge) memanggil API dan butuh beberapa detik. Tanpa ini UI diam
            # kosong sejak analis menekan kirim; untuk pertanyaan casual jedanya
            # makin terasa karena "Mencari konteks..." dilewati (hanya dikirim
            # saat technical/knowledge). Kata "Memproses" ada di THINKING_KEYS
            # (ui/index.html) sehingga tampil sebagai ThinkingBubble beranimasi
            # lalu digantikan status berikutnya — bukan chat bubble permanen.
            await websocket.send_json({"role": "bot",
                "message": "Memproses pertanyaan..."})

            valid, err = guardrails_input(data)
            if not valid:
                await websocket.send_json({"role": "bot",
                    "message": f"Input tidak valid: {err}"})
                continue

            # Klasifikasi intent + tentukan koleksi (SATU sumber kebenaran,
            # dipakai juga oleh skrip evaluasi 6a/6b via classify_intent).
            # - knowledge murni: HANYA KB domain yang relevan (MITRE/NIST), tanpa log
            # - log_analysis: butuh log + KB
            # - technical lain: fallback query semua
            intent       = classify_intent(data)
            technical    = intent["technical"]
            knowledge    = intent["knowledge"]
            log_analysis = intent["log_analysis"]
            needs_logs   = intent["needs_logs"]
            needs_nist   = intent["needs_nist"]
            needs_mitre  = intent["needs_mitre"]

            logger.info("Intent — technical=%s knowledge=%s log_analysis=%s | "
                        "needs_logs=%s needs_nist=%s needs_mitre=%s",
                        technical, knowledge, log_analysis,
                        needs_logs, needs_nist, needs_mitre)

            context = {"logs": [], "nist": [], "mitre": []}
            if technical or knowledge:
                await websocket.send_json({"role": "bot",
                    "message": "Mencari konteks di log & knowledge base..."})
                try:
                    context = retrieve_context(data,
                                               needs_logs=needs_logs,
                                               needs_nist=needs_nist,
                                               needs_mitre=needs_mitre)
                except Exception as e:
                    logger.exception("Retrieval error")
                    await websocket.send_json({"role": "bot",
                        "message": f"Gagal retrieve context: {e}"})
                    continue

            # Pilih prompt berdasarkan tipe query
            if knowledge and not log_analysis:
                # Pertanyaan faktual/konseptual — retrieve & jawab dari KB
                prompt = build_knowledge_prompt(data, context, chat_history)
            elif technical:
                # Analisis log / incident analysis — pakai full template.
                # is_knowledge DIMATIKAN saat log_analysis menyala. Pertanyaan
                # seperti "apakah ada anomali DNS dalam log? jelaskan detailnya"
                # menyalakan knowledge (gara-gara kata "jelaskan") DAN
                # log_analysis sekaligus. Tanpa penjaga ini,
                # build_technical_prompt memakai panduan konseptual yang memuat
                # "Data log TIDAK diperlukan untuk pertanyaan ini" — LLM lalu
                # mengabaikan log dan menjawab dari teori MITRE/NIST saja.
                prompt = build_technical_prompt(data, context, chat_history,
                                                is_knowledge=knowledge and not log_analysis)
            else:
                # Sapaan / pertanyaan umum
                prompt = build_casual_prompt(data, chat_history)

            await websocket.send_json({"role": "bot",
                "message": "Menyusun jawaban..."})

            try:
                raw_answer = generate_answer(prompt, technical)
            except Exception as e:
                logger.exception("LLM error")
                await websocket.send_json({"role": "bot",
                    "message": f"LLM error: {e}"})
                continue

            cleaned   = clean_output(raw_answer)
            valid_out, answer = guardrails_output(cleaned)
            if not valid_out:
                await websocket.send_json({"role": "bot",
                    "message": f"Jawaban tidak dapat diproses: {answer}"})
                continue

            # Riwayat percakapan sengaja TIDAK disimpan: tiap pertanyaan dijawab
            # independen agar topik lama (mis. DNS spoofing) tidak nyangkut ke
            # pertanyaan berikutnya.

            await websocket.send_json({"role": "bot", "message": answer})

    except WebSocketDisconnect:
        logger.info("Client disconnect.")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"role": "bot", "message": f"Error: {e}"})
        except Exception:
            pass

# ============================================================
# ROUTES
# ============================================================
UI_FILE = Path(__file__).parent / "ui" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve UI tanpa auth — login ditangani di sisi client (custom form)."""
    try:
        html = UI_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ui/index.html tidak ditemukan.")
    return html

@app.post("/api/login")
async def api_login(credentials: HTTPBasicCredentials = Depends(security)):
    """Validasi kredensial dari custom login form di UI."""
    if not verify_credentials(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah.",
        )
    return {"status": "ok", "model": LLM_MODEL}

@app.get("/health")
async def health():
    """Status endpoint tanpa auth — untuk monitoring/health check eksternal."""
    return {
        "status":       "ok",
        "logs_indexed": app_state["logs_indexed"],
        "guardrails":   "guardrails_ai_prompt_injection_detector" if _guardrails_available else "inactive",
        "model":        LLM_MODEL,
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)