# ============================================================
# 4. insert_qradar_logs_to_qdrant.py — Embed & upload log QRadar ke Qdrant
# ============================================================
# Alur: qradar_as_wazuh.jsonl (hasil script 3) -> build_text_for_embedding()
# (susun representasi teks kaya-konteks per event, termasuk flow stats CEF
# yang diekstrak sebelum full_log dipotong) -> embed (BAAI/bge-large-en-v1.5)
# -> upsert ke collection "wazuh_logs" di Qdrant — SATU collection yang sama
# dipakai log Wazuh asli dari SOCA.py (/reload), makanya skema payload dibuat
# identik dengan index_logs_to_qdrant() di SOCA.py.
# 1 event = 1 vector point. Upsert by deterministic ID (UUID dari script 3,
# atau integer offset kalau ada duplikat) — aman dijalankan ulang tanpa
# menduplikasi data, cukup menimpa payload yang sama.
# Jalankan SETELAH: python "3. convert_qradar_to_wazuh.py"
#          lalu:     python "4. insert_qradar_logs_to_qdrant.py"
# ============================================================

import os
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

load_dotenv()  # baca QDRANT_API_KEY dari .env


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
INPUT_JSONL     = "qradar_as_wazuh.jsonl"
COLLECTION_NAME = "wazuh_logs"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTOR_SIZE     = 1024
BATCH_SIZE      = 32          # jumlah event per batch upsert ke Qdrant

# Kredensial Qdrant — QDRANT_API_KEY dibaca dari .env (lihat .env.example)
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")   # isi di .env
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_HTTPS    = os.getenv("QDRANT_HTTPS", "false").lower() in ("true", "1", "yes")
QDRANT_URL      = f"{'https' if QDRANT_HTTPS else 'http'}://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")         # isi bila Qdrant butuh auth


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------

# Ekstraksi statistik flow dari RawLog CEF (Check Point/QRadar) — identik
# dengan _extract_flow_stats/_format_flow_stats di SOCA.py. Field seperti
# dst=, bytes=, connection_count=, duration= sering muncul jauh di belakang
# string CEF (>800 karakter), di luar jangkauan slice full_log yang dipotong.
# Ekstrak SEBELUM dipotong agar fakta kunci tidak hilang dari data tersimpan.
_FLOW_FIELD_PATTERNS = {
    "dst":         r"\bdst=([\d.]+)",
    "src":         r"\bsrc=([\d.]+)",
    "bytes":       r"\bbytes=(\d+)",
    "connections": r"\bconnection_count=(\d+)",
    "duration":    r"\bduration=(\d+)",
}

def extract_flow_stats(full_log: str) -> dict:
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

def format_flow_stats(stats: dict) -> str:
    """Ringkasan kompak flow stats untuk disisipkan ke teks embedding."""
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


def build_text_for_embedding(event: dict) -> str:
    """
    Membangun representasi teks dari satu event Wazuh-like untuk keperluan
    embedding. Teks dirancang agar mengandung konteks keamanan yang kaya
    sehingga retrieval semantik dapat menemukan event yang relevan secara akurat.
    """
    rule  = event.get("rule", {})
    agent = event.get("agent", {})
    data  = event.get("data", {})

    lines = [
        f"Security Event: {rule.get('description', 'Unknown')}",
        f"Timestamp: {event.get('timestamp', '')}",
        f"Agent: {agent.get('name', '')} ({agent.get('ip', '')})",
        f"Source IP: {data.get('srcip', '')}  Port: {data.get('srcport', '')}",
        f"Destination IP: {data.get('dstip', '')}  Port: {data.get('dstport', '')}",
        f"Protocol: {data.get('protocol', '')}",
        f"Rule Level: {rule.get('level', '')}  Rule ID: {rule.get('id', '')}",
        f"Groups: {', '.join(rule.get('groups', []))}",
        f"Location: {event.get('location', '')}",
        f"Action: {data.get('action', '')}",
        f"Device: {data.get('device_name', '')}  Domain: {data.get('domain', '')}",
        f"SIEM Source: {data.get('siem_source', 'qradar')}",
    ]

    # Tambahkan field Zeek jika ada (memperkaya konteks DNS/HTTP/SSL)
    zeek_fields = {k: v for k, v in data.items() if k.startswith("zeek_")}
    if zeek_fields:
        zeek_str = "  ".join(
            f"{k.replace('zeek_', '')}: {v}" for k, v in zeek_fields.items()
        )
        lines.append(f"Network Details: {zeek_str}")

    # Tambahkan field CEF jika ada (memperkaya konteks firewall)
    cef_fields = {k: v for k, v in data.items() if k.startswith("cef_")}
    if cef_fields:
        cef_str = "  ".join(
            f"{k.replace('cef_', '')}: {v}" for k, v in cef_fields.items()
        )
        lines.append(f"Firewall Details: {cef_str}")

    # Flow stats (dst/bytes/connections/duration) diekstrak sebelum full_log
    # dipotong — fakta kunci ini sering muncul jauh di belakang string CEF.
    flow_line = format_flow_stats(extract_flow_stats(event.get("full_log", "")))
    if flow_line:
        lines.append(f"Flow: {flow_line}")

    # Sertakan penggalan full_log (batas 400 karakter) untuk konteks raw
    full_log = event.get("full_log", "")
    if full_log:
        lines.append(f"Raw Log: {full_log[:400]}")

    return "\n".join(lines)


def parse_date_hour(timestamp: str) -> tuple[str, str]:
    """
    Mengurai field timestamp menjadi date ('YYYY-MM-DD') dan hour ('HH:MM:SS')
    secara terpisah. Dibutuhkan oleh _format_log_hit di SOCA.py untuk menampilkan
    waktu kejadian ke LLM.

    Format timestamp yang didukung: '2026-04-21T08:52:05.000+0700'
    Jika parsing gagal, kembalikan string kosong.
    """
    if not timestamp:
        return "", ""
    try:
        dt = datetime.fromisoformat(timestamp[:19])
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return "", ""


def load_events(jsonl_path: str) -> Generator[dict, None, None]:
    """Generator: membaca file JSONL satu baris per iterasi."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [!] Baris JSON tidak valid, dilewati: {e}")


def ensure_collection(client: QdrantClient) -> None:
    """
    Memastikan koleksi wazuh_logs sudah ada di Qdrant.
    Jika belum ada, koleksi dibuat dengan konfigurasi yang sama dengan
    koleksi yang telah digunakan pada pipeline sebelumnya
    (vector_size=1024, cosine distance).
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"[*] Koleksi '{COLLECTION_NAME}' belum ada, membuat baru...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"[+] Koleksi '{COLLECTION_NAME}' berhasil dibuat.")
    else:
        print(f"[+] Koleksi '{COLLECTION_NAME}' ditemukan, melanjutkan upsert.")


def upsert_batch(client: QdrantClient, points: list[PointStruct]) -> None:
    """Melakukan upsert satu batch point ke Qdrant."""
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


# ---------------------------------------------------------------------------
# Proses utama
# ---------------------------------------------------------------------------

def main():
    """Entry point: muat model embedding + koneksi Qdrant, cek duplikat ID,
    lalu embed & upsert seluruh event dari INPUT_JSONL per batch (BATCH_SIZE=32)."""
    # 1. Cek file input
    if not Path(INPUT_JSONL).exists():
        raise FileNotFoundError(
            f"File '{INPUT_JSONL}' tidak ditemukan. "
            "Jalankan convert_qradar_to_wazuh.py terlebih dahulu."
        )

    # 2. Muat model embedding
    print(f"[*] Memuat model embedding: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("[+] Model berhasil dimuat.")

    # 3. Koneksi ke Qdrant
    print(f"\n[*] Menghubungkan ke Qdrant: {QDRANT_URL}")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )
    ensure_collection(client)

    # 4. Baca semua event dari JSONL
    all_events = list(load_events(INPUT_JSONL))
    total = len(all_events)
    print(f"\n[*] Total event yang akan diindeks: {total}")

    # ── Deteksi duplikat ID sebelum upsert ──────────────────────────────────
    # UUID5 dari convert_qradar_to_wazuh.py bisa identik jika field sumber
    # (qid + startTime + srcip + dstport) sama, maka pakai integer offset
    # dari jumlah point yang sudah ada agar tidak tabrakan dengan Wazuh logs.
    original_ids = [e.get("id", "") for e in all_events]
    unique_ids   = set(original_ids)
    if len(unique_ids) < total:
        print(f"  [!] Ditemukan {total - len(unique_ids)} duplikat UUID — "
              f"menggunakan integer offset sebagai point ID.")
        use_index_as_id = True
    else:
        print(f"  [i] Semua {total} event memiliki UUID unik.")
        use_index_as_id = False

    # Ambil jumlah point yang sudah ada untuk menghitung offset integer ID.
    # Ini mencegah tabrakan dengan QRadar batch sebelumnya kalau koleksi
    # sudah berisi data Wazuh (UUID string) + QRadar (integer) lama.
    try:
        existing_count = client.get_collection(COLLECTION_NAME).points_count or 0
    except Exception:
        existing_count = 0
    # ────────────────────────────────────────────────────────────────────────

    # 5. Proses per batch
    inserted   = 0
    batch_num  = 0
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_events = all_events[batch_start: batch_start + BATCH_SIZE]
        batch_num += 1

        texts    = [build_text_for_embedding(e) for e in batch_events]
        # PERHATIAN: prefix ini HANYA dipakai di sini. SOCA.py (baik saat encode
        # query maupun saat index_logs_to_qdrant() meng-embed log Wazuh asli)
        # TIDAK memakai prefix apa pun. Akibatnya vector event QRadar di
        # collection wazuh_logs berada di "ruang representasi" yang sedikit
        # berbeda dari vector query & log Wazuh — berpotensi menurunkan skor
        # cosine similarity untuk event QRadar secara sistematis. Belum
        # diverifikasi seberapa besar dampaknya terhadap kualitas retrieval.
        prefixed = [f"Represent this sentence: {t}" for t in texts]
        vectors  = model.encode(prefixed, normalize_embeddings=True).tolist()

        points = []
        for local_idx, (event, text, vector) in enumerate(
            zip(batch_events, texts, vectors)
        ):
            if use_index_as_id:
                # Integer unik: offset dari existing_count agar tidak bertabrakan
                point_id = existing_count + batch_start + local_idx
            else:
                point_id = event["id"]   # UUID string asli

            # --- FIX: parse date dan hour dari timestamp ---
            # Field ini dibutuhkan oleh _format_log_hit di SOCA.py.
            # Tanpa field ini, LLM menerima "- -" sebagai timestamp dan
            # menyimpulkan "tidak ada informasi tentang waktu dan tanggal".
            ts_raw = event.get("timestamp", "")
            date_str, hour_str = parse_date_hour(ts_raw)
            # -----------------------------------------------

            payload = {
                # Skema identik dengan index_logs_to_qdrant() di SOCA.py (/reload)
                "text":            text,
                "timestamp":       ts_raw,
                "date":            date_str,
                "hour":            hour_str,
                "rule_id":         str(event.get("rule", {}).get("id", "")),
                "rule_level":      int(event.get("rule", {}).get("level", 0) or 0),
                "rule_desc":       event.get("rule", {}).get("description", ""),
                "rule_groups":     ", ".join(event.get("rule", {}).get("groups", [])),
                "rule_fired":      1,
                "agent_name":      event.get("agent", {}).get("name", ""),
                "agent_id":        str(event.get("agent", {}).get("id", "")),
                "agent_ip":        event.get("agent", {}).get("ip", ""),
                "mitre_id":        "",
                "mitre_tactic":    "",
                "mitre_technique": "",
                "srcip":           event.get("data", {}).get("srcip", ""),
                "srcport":         str(event.get("data", {}).get("srcport", "")),
                "srcuser":         event.get("data", {}).get("srcuser", ""),
                "dstuser":         event.get("data", {}).get("dstuser", ""),
                "protocol":        event.get("data", {}).get("protocol", ""),
                "url":             (event.get("data", {}).get("zeek_uri", "")
                                    or event.get("data", {}).get("url", "")),
                "command":         event.get("data", {}).get("command", ""),
                "location":        event.get("location", ""),
                "decoder":         "",
                "full_log":        (event.get("full_log", "") or "")[:1500],
                "flow_stats":      extract_flow_stats(event.get("full_log", "")),
                "pci_dss":         "",
                "nist_800_53":     "",
                "source":          "qradar_log",
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        upsert_batch(client, points)
        inserted += len(points)

        elapsed = time.time() - start_time
        print(f"  Batch {batch_num:3d} | {inserted:4d}/{total} event | {elapsed:.1f}s")

    # 6. Verifikasi akhir
    collection_info = client.get_collection(COLLECTION_NAME)
    final_count     = collection_info.points_count

    print(f"\n[+] Selesai! {inserted} event berhasil diupsert.")
    print(f"    Total point di koleksi '{COLLECTION_NAME}': {final_count}")
    print(f"    Durasi total: {time.time() - start_time:.1f} detik")

    # Konfirmasi
    if final_count >= inserted:
        print(f"    [✓] Semua {total} event berhasil disimpan.")
    else:
        print(f"    [!] Jumlah tidak sesuai ekspektasi: {final_count} point di koleksi.")


if __name__ == "__main__":
    main()