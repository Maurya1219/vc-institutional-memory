"""
Incremental Affinity list sync: diff raw list entries vs cache, re-enrich only
changed rows, merge caches, then rebuild FAISS from all caches when anything changed.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ingest_affinity_full import (
    LISTS_CONFIG,
    build_company_doc,
    build_contact_doc,
    cache_path,
    embed_with_retry,
    embeddings,
    enrich_single_entry,
    get_all_pages,
)

load_dotenv()

API_KEY = os.getenv("AFFINITY_API_KEY")
CACHE_DIR = Path("affinity_cache")
SYNC_LOG = CACHE_DIR / "sync_log.json"

CACHE_DIR.mkdir(exist_ok=True)


def load_cache_entries(name: str) -> list:
    path = cache_path(name)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def write_list_cache(name: str, data: list) -> None:
    path = cache_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cached_at": datetime.now().isoformat(), "data": data}, f)


def strip_affinity(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "_affinity"}


def entry_hash(entry: dict) -> str:
    return json.dumps(strip_affinity(entry), sort_keys=True, default=str)


def get_entry_key(entry: dict) -> str:
    return str(entry.get("id") or entry.get("entity_id", ""))


def enrich_one(entry: dict, list_config: dict) -> dict:
    return enrich_single_entry(entry, list_config)


def merge_list_entries(
    list_config: dict, old_entries: list, new_raw: list
) -> tuple[list, dict]:
    """Return merged enriched list + counts added/modified/removed."""
    name = list_config["name"]
    old_by_key = {get_entry_key(e): e for e in old_entries if get_entry_key(e)}
    old_hash = {k: entry_hash(e) for k, e in old_by_key.items()}

    new_keys = set()
    merged: list = []
    n_added = n_modified = 0

    for entry in new_raw:
        k = get_entry_key(entry)
        if not k:
            continue
        new_keys.add(k)
        nh = entry_hash(entry)
        if k not in old_by_key:
            merged.append(enrich_one(entry, list_config))
            n_added += 1
            time.sleep(0.05)
        elif nh != old_hash.get(k):
            merged.append(enrich_one(entry, list_config))
            n_modified += 1
            time.sleep(0.05)
        else:
            merged.append(old_by_key[k])

    removed = len(set(old_by_key.keys()) - new_keys)

    return merged, {
        "list": name,
        "added": n_added,
        "modified": n_modified,
        "removed": removed,
    }


def rebuild_vectorstore_from_caches() -> int:
    # Replaces dvc_vectorstore with Affinity-derived docs only (not SharePoint merges).
    all_docs = []
    for list_config in LISTS_CONFIG:
        name = list_config["name"]
        entries = load_cache_entries(name)
        is_contacts = list_config["id"] == 237644
        for entry in tqdm(entries, desc=f"Docs {name[:20]}"):
            try:
                if is_contacts:
                    doc = build_contact_doc(entry, name)
                else:
                    doc = build_company_doc(entry, name)
                if doc:
                    all_docs.append(doc)
            except Exception:
                pass

    if not all_docs:
        return 0

    batch_size = 500
    first = all_docs[:batch_size]
    vectorstore = embed_with_retry(first, embeddings)
    if len(all_docs) > batch_size:
        for i in tqdm(range(batch_size, len(all_docs), batch_size)):
            batch = all_docs[i : i + batch_size]
            vectorstore.add_documents(batch)

    vectorstore.save_local("dvc_vectorstore")
    return len(all_docs)


def log_sync(summary: dict) -> None:
    log = []
    if SYNC_LOG.exists():
        with open(SYNC_LOG, encoding="utf-8") as f:
            log = json.load(f)
    log.append({"timestamp": datetime.now().isoformat(), "results": summary})
    log = log[-30:]
    with open(SYNC_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def run_sync() -> list:
    if API_KEY in (None, "", "your-key-here"):
        raise RuntimeError("AFFINITY_API_KEY is not configured.")

    print(f"\nStarting Affinity sync — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    total_added = total_modified = total_removed = 0
    sync_results: list = []
    any_change = False

    for list_config in LISTS_CONFIG:
        name = list_config["name"]
        list_id = list_config["id"]
        max_results = list_config.get("max")

        print(f"\nChecking {name}...")
        old_entries = load_cache_entries(name)
        new_raw = get_all_pages(
            f"lists/{list_id}/list-entries",
            key="list_entries",
            max_results=max_results,
        )

        merged, stats = merge_list_entries(list_config, old_entries, new_raw)
        write_list_cache(name, merged)

        print(
            f"  Added: {stats['added']} | Modified: {stats['modified']} | "
            f"Removed: {stats['removed']}"
        )

        total_added += stats["added"]
        total_modified += stats["modified"]
        total_removed += stats["removed"]
        sync_results.append(stats)

        if stats["added"] or stats["modified"] or stats["removed"]:
            any_change = True

    doc_count = 0
    if any_change:
        print("\nChanges detected — rebuilding vector store from merged caches...")
        doc_count = rebuild_vectorstore_from_caches()
        print(f"Vector store rebuilt ({doc_count} documents).")
        try:
            from memory_engine import reload_vectorstore

            reload_vectorstore()
        except Exception as e:
            print(f"Could not reload in-memory vector store: {e}")
    else:
        print("\nNo list entry changes — vector store unchanged.")

    summary = {
        "total_added": total_added,
        "total_modified": total_modified,
        "total_removed": total_removed,
        "lists": sync_results,
        "documents_indexed": doc_count,
    }
    log_sync(summary)

    print(
        f"\nSync complete — {total_added} added, {total_modified} modified, "
        f"{total_removed} removed"
    )
    return sync_results


if __name__ == "__main__":
    run_sync()
