# ============================================================
# 3. convert_qradar_to_wazuh.py — Konversi CSV export QRadar ke format Wazuh
# ============================================================
# Alur: CSV export QRadar -> buang event 'Health Metric' (log internal QRadar,
# bukan event keamanan) -> build_wazuh_event() per baris (parse timestamp,
# map severity QRadar 0-10 ke rule level Wazuh 0-15, ekstrak data Zeek/CEF
# tertanam di payload, bangun rule.groups dari kategori/nama event) -> tulis
# 1 objek JSON per baris ke qradar_as_wazuh.jsonl (format JSONL, menyerupai
# struktur archives.json Wazuh asli).
# Output JSONL ini yang di-embed & di-upload ke Qdrant oleh script 4 — proses
# ini sendiri TIDAK menyentuh Qdrant, murni transformasi format data lokal.
# Jalankan: python "3. convert_qradar_to_wazuh.py"
# ============================================================

import pandas as pd
import json
import re
import uuid
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
# CATATAN: INPUT_CSV pakai path relatif — file harus ada di folder yang sama
# dengan script ini saat dijalankan. Output (qradar_as_wazuh.jsonl) di root
# proyek SUDAH merupakan hasil konversi yang dipakai pipeline (script 4,
# ground truth, RAGAS eval) — script ini tidak perlu dijalankan ulang kecuali
# ingin regenerasi dari CSV mentah.
INPUT_CSV    = "2026-04-21-data_export.csv"
OUTPUT_JSONL = "qradar_as_wazuh.jsonl"

# Event internal QRadar yang tidak relevan untuk analisis keamanan
EXCLUDED_EVENTS = {"Health Metric"}

# Zona waktu WIB (UTC+7)
WIB = timezone(timedelta(hours=7))

# Mapping logSourceIdentifier → agent_id Wazuh (format "001", "002", dst.)
AGENT_ID_MAP = {
    "kmtn-zeek-core": "010",
    "kmtn-zeek-dmz":  "011",
    "10.1.233.14":    "012",
}


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------

def map_severity_to_wazuh_level(qradar_severity: float) -> int:
    """
    Memetakan severity QRadar (skala 0–10) ke level rule Wazuh (skala 0–15).
    QRadar menggunakan skala 10 poin, sementara Wazuh menggunakan 15 poin.
    """
    # Mapping non-linear (bukan sekadar dikali 1.5) — beberapa titik severity
    # QRadar (mis. 6->9, bukan 6->9.5 dibulatkan) disesuaikan manual supaya
    # ambang MIN_RULE_LEVEL=5 di SOCA.py tetap masuk akal untuk data QRadar.
    mapping = {
        0: 0, 1: 2, 2: 3, 3: 5, 4: 6,
        5: 7, 6: 9, 7: 10, 8: 12, 9: 13, 10: 15
    }
    try:
        return mapping.get(int(float(qradar_severity)), 5)
    except (ValueError, TypeError):
        return 5


def parse_timestamp(dt_str: str) -> str:
    """
    Mengurai string datetime QRadar ('Apr 21, 2026, 8:52:05 AM') ke format
    timestamp Wazuh ('2026-04-21T08:52:05.000+0700').
    Jika parsing gagal, kembalikan timestamp saat ini.
    """
    # Coba beberapa format karena export QRadar tidak selalu konsisten
    # (kadang tanpa detik, kadang sudah ISO kalau data sudah diproses ulang).
    formats = [
        "%b %d, %Y, %I:%M:%S %p",
        "%b %d, %Y, %I:%M %p",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(str(dt_str).strip(), fmt)
            dt = dt.replace(tzinfo=WIB)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.000+0700")
        except ValueError:
            continue
    return datetime.now(tz=WIB).strftime("%Y-%m-%dT%H:%M:%S.000+0700")


def extract_zeek_data(payload: str) -> dict:
    """
    Mengekstrak data Zeek JSON yang tertanam dalam payload syslog.
    Format payload: '<priority>MMM DD HH:MM:SS hostname zeek: {JSON}'
    """
    if not isinstance(payload, str):
        return {}
    match = re.search(r'zeek:\s*(\{.*\})\s*$', payload, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def parse_cef_data(payload: str) -> dict:
    """
    Mengurai log berformat CEF (Check Point / firewall) dari payload.
    Format: 'CEF:0|vendor|product|version|sig_id|name|severity|extensions'
    Mengembalikan dict berisi pasangan key=value dari bagian ekstensi CEF.
    """
    if not isinstance(payload, str) or not payload.startswith("CEF:"):
        return {}
    data = {}
    parts = payload.split("|", 7)
    if len(parts) >= 8:
        extension = parts[7]
        pairs = re.findall(r'(\w+)=(.*?)(?=\s+\w+=|$)', extension)
        for key, value in pairs:
            data[key.strip()] = value.strip()
    return data


def build_rule_groups(category_desc: str, event_name: str, log_type: str) -> list:
    """
    Membangun daftar grup rule Wazuh berdasarkan kategori dan nama event.
    QRadar tidak punya taksonomi rule_groups seperti Wazuh, jadi grup di sini
    disimpulkan dari keyword matching pada nama/kategori event QRadar — bukan
    field asli dari sumbernya (beda dengan rule_id/level yang memang ada datanya).
    """
    groups = []
    category_lower = str(category_desc).lower()
    event_lower    = str(event_name).lower()

    if "firewall" in category_lower or "permit" in category_lower or "accept" in event_lower:
        groups.extend(["firewall", "allow"])
    if "dns" in category_lower or "dns" in event_lower:
        groups.extend(["dns", "network"])
    if "http" in event_lower:
        groups.extend(["web", "http"])
    if "ssl" in event_lower or "https" in event_lower:
        groups.extend(["web", "ssl"])
    if "ssh" in event_lower:
        groups.extend(["sshd", "remote_access"])
    if "icmp" in event_lower:
        groups.extend(["network", "icmp"])
    if "vpn" in event_lower:
        groups.extend(["vpn", "firewall"])
    if "spoof" in event_lower:
        groups.extend(["attack", "spoofing"])
    if "attack" in event_lower or "suspicious" in event_lower or "ips" in event_lower:
        groups.extend(["attack", "intrusion_detection"])
    if log_type:
        groups.append(log_type)

    groups.append("qradar")

    # Hilangkan duplikat sambil mempertahankan urutan
    seen = set()
    return [g for g in groups if not (g in seen or seen.add(g))]


def safe_str(value, default: str = "") -> str:
    """Konversi nilai ke string; kembalikan default jika NaN/None."""
    if value is None:
        return default
    # Cek NaN hanya untuk tipe non-string (pd.isna melempar error pada string)
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def build_wazuh_event(row: pd.Series) -> dict:
    """
    Membangun satu objek JSON menyerupai format arsip Wazuh dari satu baris
    QRadar. Struktur disesuaikan dengan Wazuh archives.json agar kompatibel
    dengan pipeline RAG yang sudah ada.
    """
    # --- Timestamp ---
    timestamp = parse_timestamp(row.get("deviceDateTime", ""))

    # --- Rule ---
    severity      = row.get("severity", 3)
    wazuh_level   = map_severity_to_wazuh_level(severity)
    event_name    = safe_str(row.get("eventName"), "Unknown Event")
    category_desc = safe_str(row.get("categoryDescription"))
    qid           = safe_str(row.get("qid"), "0")

    # --- Agent ---
    log_source = safe_str(row.get("logSourceIdentifier"), "unknown")
    src_ip     = safe_str(row.get("srcIp"), "0.0.0.0")
    # "099" untuk log source yang belum terpetakan di AGENT_ID_MAP — sengaja
    # beda dari "000" (konvensi Wazuh untuk manager internal) supaya tidak
    # ikut ke-filter oleh ENDPOINT_ONLY di SOCA.py.
    agent_id   = AGENT_ID_MAP.get(log_source, "099")

    # Jika logSourceIdentifier adalah IP, jadikan sebagai agent IP
    agent_ip = log_source if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', log_source) else src_ip

    # --- Payload dan data tambahan ---
    payload  = safe_str(row.get("payloadAsUTF"))
    zeek     = extract_zeek_data(payload)
    cef      = parse_cef_data(payload)

    # Tentukan log_type dari Zeek atau dari nama event
    log_type = zeek.get("log_type", "")
    if not log_type:
        event_lower = event_name.lower()
        if "dns" in event_lower:
            log_type = "dns"
        elif "http" in event_lower:
            log_type = "http"
        elif "ssl" in event_lower:
            log_type = "ssl"
        elif any(kw in event_lower for kw in ("firewall", "accept", "permit")):
            log_type = "conn"

    # --- Bangun data field ---
    data = {
        "srcip":       src_ip,
        "dstip":       safe_str(row.get("destIp"), "0.0.0.0"),
        "srcport":     safe_str(row.get("sourcePort"), "0"),
        "dstport":     safe_str(row.get("destinationPort"), "0"),
        "protocol":    safe_str(row.get("protocolName")),
        "action":      event_name,
        "severity":    safe_str(severity),
        "magnitude":   safe_str(row.get("magnitude")),
        "device_name": safe_str(row.get("deviceName")),
        "domain":      safe_str(row.get("domainName")),
        "log_source":  log_source,
        "event_id":    safe_str(row.get("deviceEventId")),
        "rule_ids":    safe_str(row.get("ruleIds")),
        "siem_source": "qradar",
    }

    # Tambahkan field Zeek jika tersedia
    if zeek:
        for key in ["query", "method", "status_code", "uri", "resp_mime_types",
                    "server_name", "uid", "proto"]:
            if key in zeek:
                data[f"zeek_{key}"] = str(zeek[key])

    # Tambahkan field CEF jika tersedia
    if cef:
        for key in ["act", "spt", "dpt", "cs2", "layer_name", "rule_uid",
                    "sourceTranslatedAddress"]:
            if key in cef:
                data[f"cef_{key}"] = cef[key]

    # UUID deterministik berdasarkan konten event untuk deduplication
    event_uuid = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{qid}-{safe_str(row.get('startTime'))}-{src_ip}-{safe_str(row.get('destinationPort'))}",
    ))

    wazuh_event = {
        "timestamp": timestamp,
        "rule": {
            "level":       wazuh_level,
            "description": event_name,
            "id":          qid,
            "groups":      build_rule_groups(category_desc, event_name, log_type),
        },
        "agent": {
            "id":   agent_id,
            "name": log_source,
            "ip":   agent_ip,
        },
        "manager": {
            "name": safe_str(row.get("deviceName"), "KMTN-QRadar"),
        },
        "id":       event_uuid,
        # 2000 char cukup longgar di sini — pemotongan LEBIH KETAT terjadi lagi
        # di script 4 ([:1500]) dan di SOCA.py saat format ke prompt ([:400]).
        # Field flow (dst/bytes/connections/duration) diekstrak terpisah SEBELUM
        # potongan-potongan itu supaya tidak hilang meski full_log akhirnya pendek.
        "full_log": payload[:2000] if payload else "",
        "location": f"qradar/{log_type or 'syslog'}",
        "data":     data,
    }

    return wazuh_event


# ---------------------------------------------------------------------------
# Proses utama
# ---------------------------------------------------------------------------

def main():
    """Entry point: baca INPUT_CSV, buang event 'Health Metric', konversi
    baris sisanya satu-per-satu via build_wazuh_event(), tulis ke OUTPUT_JSONL
    (1 baris JSON per event). Error per-baris di-skip (tidak menghentikan proses)."""
    print(f"[*] Membaca file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    total_awal = len(df)

    # Filter event internal QRadar
    df = df[~df["eventName"].isin(EXCLUDED_EVENTS)]
    total_setelah_filter = len(df)
    print(f"[*] Total baris awal        : {total_awal}")
    print(f"[*] Setelah filter excluded : {total_setelah_filter} baris")

    converted = 0
    errors    = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            try:
                event = build_wazuh_event(row)
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                converted += 1
            except Exception as e:
                errors += 1
                print(f"  [!] Error baris {idx}: {e}")

    print(f"\n[+] Konversi selesai.")
    print(f"    Berhasil dikonversi : {converted} event")
    print(f"    Error               : {errors} baris")
    print(f"    Output              : {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
