# ============================================================
# 1. mitre_upload.py — Upload knowledge base MITRE ATT&CK ke Qdrant
# ============================================================
# Alur: mitre_attack.json (STIX 2.1 bundle resmi MITRE) -> build_relation_maps()
# (petakan relasi mitigasi/deteksi/parent-child antar objek STIX) ->
# parse_mitre_attack() (1 technique/sub-technique = 1 chunk teks) -> embed
# (BAAI/bge-large-en-v1.5) -> upsert ke collection "mitre_attack" di Qdrant.
# Full refresh: collection lama di-drop lalu dibuat ulang tiap dijalankan
# (beda dengan wazuh_logs yang upsert incremental).
# Jalankan sekali saat setup, atau ulang kalau mitre_attack.json diperbarui.
# ============================================================

import os
import sys
import json
import uuid
import logging
import warnings

from dotenv import load_dotenv

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
logger = logging.getLogger("mitre_upload")

# ============================================================
# KONFIGURASI
# Secret (QDRANT_API_KEY) dibaca dari .env — lihat .env.example
# ============================================================
load_dotenv()

QDRANT_HOST    = os.getenv("QDRANT_HOST", "")            # WAJIB: IP/hostname server Qdrant (isi di .env)
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")         # isi bila Qdrant butuh auth
COLLECTION     = "mitre_attack"
VECTOR_SIZE    = 1024
BATCH_SIZE     = 8       # chunk MITRE besar; kecilkan untuk hindari timeout
TIMEOUT        = 120     # detik

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
MITRE_FILE  = "mitre_attack.json"

# Batas agar chunk tidak terlalu panjang untuk embedding model (512-token limit)
MAX_MITIGATIONS = 5
MAX_ANALYTICS   = 3
MAX_LOG_SOURCES = 4


# ============================================================
# RELATION MAPS
# ============================================================
def build_relation_maps(data: dict) -> dict:
    """
    Bangun peta relasi dari STIX bundle:
    - objects_by_id       : stix_id -> object
    - tech_to_mitigations : stix_id -> [(mitigation_obj, rel_description)]
    - tech_to_det_strats  : stix_id -> [detection_strategy_obj]
    - child_to_parent     : subtechnique stix_id -> parent stix_id
    - analytics_by_id     : stix_id -> x-mitre-analytic object
    """
    objects_by_id = {o["id"]: o for o in data["objects"]}

    tech_to_mitigations: dict[str, list] = {}
    tech_to_det_strats:  dict[str, list] = {}
    child_to_parent:     dict[str, str]  = {}

    for obj in data["objects"]:
        if obj.get("type") != "relationship":
            continue

        rel_type = obj.get("relationship_type")
        src_ref  = obj.get("source_ref", "")
        tgt_ref  = obj.get("target_ref", "")
        rel_desc = obj.get("description", "")

        if rel_type == "mitigates":
            src_obj = objects_by_id.get(src_ref)
            if src_obj and not src_obj.get("x_mitre_deprecated", False):
                tech_to_mitigations.setdefault(tgt_ref, []).append((src_obj, rel_desc))

        elif rel_type == "detects":
            src_obj = objects_by_id.get(src_ref)
            if src_obj and not src_obj.get("x_mitre_deprecated", False):
                tech_to_det_strats.setdefault(tgt_ref, []).append(src_obj)

        elif rel_type == "subtechnique-of":
            child_to_parent[src_ref] = tgt_ref

    analytics_by_id = {
        o["id"]: o for o in data["objects"]
        if o.get("type") == "x-mitre-analytic"
    }

    return {
        "objects_by_id":       objects_by_id,
        "tech_to_mitigations": tech_to_mitigations,
        "tech_to_det_strats":  tech_to_det_strats,
        "child_to_parent":     child_to_parent,
        "analytics_by_id":     analytics_by_id,
    }


# ============================================================
# FORMAT HELPERS
# ============================================================
def _clean(text: str, max_len: int = 0) -> str:
    """Strip whitespace, potong ke max_len karakter + '...' kalau kepanjangan (0 = tanpa batas)."""
    t = (text or "").strip()
    if max_len and len(t) > max_len:
        t = t[:max_len] + "..."
    return t


def _build_mitigations_block(stix_id: str, maps: dict) -> tuple[str, list]:
    """Bangun blok teks mitigasi + daftar nama mitigasi untuk payload."""
    entries = maps["tech_to_mitigations"].get(stix_id, [])
    lines, names = [], []

    for mit_obj, rel_desc in entries[:MAX_MITIGATIONS]:
        name     = mit_obj.get("name", "")
        mit_desc = _clean(mit_obj.get("description", ""), 400)
        ctx      = _clean(rel_desc, 200)

        block = f"- {name}: {mit_desc}"
        if ctx:
            block += f" | Context: {ctx}"
        lines.append(block)
        names.append(name)

    remaining = len(entries) - MAX_MITIGATIONS
    if remaining > 0:
        lines.append(f"  (dan {remaining} mitigasi lainnya — lihat https://attack.mitre.org)")

    return "\n".join(lines), names


def _build_detection_block(stix_id: str, maps: dict) -> tuple[str, list]:
    """Bangun blok deteksi dari detection strategies + analytics-nya."""
    strategies  = maps["tech_to_det_strats"].get(stix_id, [])
    lines, log_src_list = [], []

    for ds in strategies[:MAX_ANALYTICS]:
        analytic_refs = ds.get("x_mitre_analytic_refs", [])
        for ar_id in analytic_refs[:2]:
            an = maps["analytics_by_id"].get(ar_id)
            if not an or an.get("x_mitre_deprecated"):
                continue

            an_desc     = _clean(an.get("description", ""), 400)
            an_plat     = ", ".join(an.get("x_mitre_platforms", []))
            log_sources = an.get("x_mitre_log_source_references", [])

            lines.append(f"  [{an.get('name', '')}] (Platform: {an_plat})")
            if an_desc:
                lines.append(f"  {an_desc}")

            src_names = []
            for ls in log_sources[:MAX_LOG_SOURCES]:
                ls_name = ls.get("name", "")
                ls_chan = ls.get("channel", "")
                if ls_name:
                    entry = ls_name
                    if ls_chan:
                        entry += f" ({ls_chan})"
                    src_names.append(entry)
            if src_names:
                lines.append(f"  Log Sources: {' | '.join(src_names)}")
                log_src_list.extend(src_names)

    return "\n".join(lines), list(set(log_src_list))


# ============================================================
# PARSER — MITRE ATT&CK JSON (Technique chunks)
# ============================================================
def parse_mitre_attack(filepath: str) -> list[dict]:
    """
    Ubah 1 objek STIX "attack-pattern" (technique/sub-technique) jadi 1 chunk teks
    siap-embed, dilengkapi mitigasi dan analitik deteksi hasil join relation maps.
    Filter: buang teknik deprecated/revoked. Field payload penting untuk retrieval
    di SOCA.py: technique_id (exact-match di _hybrid_search), name, tactics,
    mitigation_names, log_sources.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    maps = build_relation_maps(data)

    # Hanya ambil teknik aktif (tidak deprecated, tidak revoked)
    techniques = [
        o for o in data["objects"]
        if o.get("type") == "attack-pattern"
        and not o.get("x_mitre_deprecated", False)
        and not o.get("revoked", False)
    ]

    chunks = []

    for tech in techniques:
        # Identitas teknik dari external_references
        technique_id = ""
        url          = ""
        for ref in tech.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id", "")
                url          = ref.get("url", "")
                break

        name        = tech.get("name", "")
        description = _clean(tech.get("description", ""), 800)
        tactics     = [p["phase_name"] for p in tech.get("kill_chain_phases", [])]
        platforms   = tech.get("x_mitre_platforms", [])
        is_sub      = tech.get("x_mitre_is_subtechnique", False)

        # Nama parent kalau sub-technique
        parent_name = ""
        if is_sub:
            parent_stix = maps["child_to_parent"].get(tech["id"])
            if parent_stix:
                parent_obj  = maps["objects_by_id"].get(parent_stix, {})
                parent_name = parent_obj.get("name", "")

        # Legacy detection field (lebih lama, tetap disertakan sebagai fallback)
        legacy_detection = _clean(tech.get("x_mitre_detection", ""), 400)

        # Mitigasi
        mitigation_block, mitigation_names = _build_mitigations_block(tech["id"], maps)

        # Deteksi dari analytics
        detection_block, log_sources = _build_detection_block(tech["id"], maps)

        # Bangun teks chunk (info penting di atas untuk ranking retrieval)
        lines = []
        tech_type = "Sub-technique" if is_sub else "Technique"
        lines.append(f"MITRE ATT&CK {tech_type}: {technique_id} — {name}")
        if parent_name:
            lines.append(f"Parent Technique: {parent_name}")
        lines.append(f"Tactics: {', '.join(tactics)}")
        lines.append(f"Platforms: {', '.join(platforms)}")
        lines.append(f"Reference: {url}")
        lines.append("\nDescription:")
        lines.append(description)

        if mitigation_block:
            lines.append("\nMitigations:")
            lines.append(mitigation_block)

        if detection_block:
            lines.append("\nDetection Analytics:")
            lines.append(detection_block)
        elif legacy_detection:
            lines.append("\nDetection Guidance:")
            lines.append(legacy_detection)

        text = "\n".join(lines).strip()

        chunks.append({
            "id":               str(uuid.uuid4()),
            "text":             text,
            "source":           "mitre_attack",
            "technique_id":     technique_id,
            "name":             name,
            "tactics":          tactics,
            "platforms":        platforms,
            "is_subtechnique":  is_sub,
            "parent_technique": parent_name,
            "has_mitigations":  bool(mitigation_block),
            "has_detection":    bool(detection_block or legacy_detection),
            "mitigation_names": mitigation_names,
            "log_sources":      log_sources[:10],
            "url":              url,
        })

    return chunks


# ============================================================
# INDEX KE QDRANT (dengan retry)
# ============================================================
def index_chunks(chunks: list[dict], qdrant: QdrantClient,
                 embedder: SentenceTransformer):
    """Embed + upsert chunks ke Qdrant per batch (BATCH_SIZE=8, kecil karena
    chunk MITRE panjang). Retry maks 3x per batch kalau timeout/error jaringan."""
    logger.info("Indexing %d chunks ke Qdrant (batch=%d)...",
                len(chunks), BATCH_SIZE)

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="MITRE ATT&CK"):
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

        # Retry maks 3 kali kalau timeout
        for attempt in range(3):
            try:
                qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning("  [Retry %d/3] Error pada batch %d: %s",
                                   attempt + 1, i // BATCH_SIZE + 1, e)
                else:
                    raise RuntimeError(
                        f"Gagal upsert setelah 3 percobaan: {e}"
                    ) from e


# ============================================================
# MAIN
# ============================================================
def main(filepath: str = MITRE_FILE):
    """Entry point: drop+recreate collection mitre_attack (full refresh), parse
    mitre_attack.json, lalu index seluruh chunk. Jalankan: python "1. mitre_upload.py"."""
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting ke Qdrant %s:%s", QDRANT_HOST, QDRANT_PORT)
    qdrant = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT,
        api_key=QDRANT_API_KEY, https=False, timeout=TIMEOUT,
    )

    # Drop & recreate collection (full refresh)
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
        logger.info("Collection lama '%s' dihapus.", COLLECTION)

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Collection '%s' dibuat.", COLLECTION)

    logger.info("Parsing %s...", filepath)
    chunks = parse_mitre_attack(filepath)
    logger.info("Total technique chunks: %d", len(chunks))

    # Statistik kualitas data
    with_mit = sum(1 for c in chunks if c["has_mitigations"])
    with_det = sum(1 for c in chunks if c["has_detection"])
    subtechs = sum(1 for c in chunks if c["is_subtechnique"])
    logger.info("  Sub-techniques  : %d", subtechs)
    logger.info("  Parent techniques: %d", len(chunks) - subtechs)
    if chunks:
        logger.info("  Dengan mitigasi : %d (%d%%)",
                    with_mit, with_mit * 100 // len(chunks))
        logger.info("  Dengan deteksi  : %d (%d%%)",
                    with_det, with_det * 100 // len(chunks))

    index_chunks(chunks, qdrant, embedder)

    info = qdrant.get_collection(COLLECTION)
    logger.info("Selesai. Total vectors di Qdrant '%s': %d",
                COLLECTION, info.points_count)


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else MITRE_FILE
    main(filepath)
