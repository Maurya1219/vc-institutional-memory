import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("AFFINITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTH = ("", API_KEY)
BASE_URL = "https://api.affinity.co"
CACHE_DIR = Path("affinity_cache")
CACHE_DIR.mkdir(exist_ok=True)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

ENTITY_TYPE_OPPORTUNITY = 8

LISTS_CONFIG = [
    {"id": 237676, "name": "Master DVC Advantage Tracker", "priority": "high", "max": None},
    {"id": 228868, "name": "DVC US Pipeline", "priority": "high", "max": None},
    {"id": 228823, "name": "DVC VC Ecosystem Interactions", "priority": "high", "max": None},
    {"id": 238067, "name": "DVC India Pipeline", "priority": "medium", "max": None},
    {"id": 237644, "name": "DVC Contacts 4D", "priority": "medium", "max": 2000},
]


def embed_with_retry(docs, embeddings, retries=3):
    for attempt in range(retries):
        try:
            return FAISS.from_documents(docs, embeddings)
        except Exception as e:
            if attempt < retries - 1:
                print(
                    f"Embedding attempt {attempt + 1} failed: {e}. Retrying in 10s..."
                )
                time.sleep(10)
            else:
                raise


def get(endpoint, params=None):
    resp = requests.get(
        f"{BASE_URL}/{endpoint}",
        auth=AUTH,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_all_pages(endpoint, key=None, params=None, max_results=None):
    results = []
    page_token = None
    while True:
        p = params.copy() if params else {}
        p["page_size"] = 100
        if page_token:
            p["page_token"] = page_token
        data = get(endpoint, params=p)
        if isinstance(data, dict) and key:
            items = data.get(key, data)
        else:
            items = data
        if isinstance(items, list):
            results.extend(items)
        if max_results is not None and len(results) >= max_results:
            results = results[:max_results]
            break
        next_token = data.get("next_page_token") if isinstance(data, dict) else None
        if not next_token:
            break
        page_token = next_token
        time.sleep(0.15)
    return results


def cache_path(name):
    safe = name.replace(" ", "_").lower()
    return CACHE_DIR / f"{safe}.json"


def save_cache(name, data):
    path = cache_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cached_at": datetime.now().isoformat(), "data": data}, f)
    print(f"Cached {len(data)} items → {path}")


def load_cache(name):
    p = cache_path(name)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    print(f"Loaded {len(obj['data'])} items from cache ({obj['cached_at'][:10]})")
    return obj["data"]


def cache_is_fresh(name, max_days=7):
    p = cache_path(name)
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    raw = obj["cached_at"].replace("Z", "+00:00")
    try:
        cached_at = datetime.fromisoformat(raw)
    except ValueError:
        cached_at = datetime.fromisoformat(obj["cached_at"][:19])
    if cached_at.tzinfo is not None:
        cached_at = cached_at.replace(tzinfo=None)
    age = datetime.now() - cached_at
    return age.days < max_days


def fetch_org(org_id):
    try:
        return get(f"organizations/{org_id}")
    except Exception:
        return {}


def fetch_opportunity(opportunity_id):
    try:
        return get(f"opportunities/{opportunity_id}")
    except Exception:
        return {}


def fetch_notes_for_entry(entry):
    org_ids = entry.get("organization_ids") or []
    person_ids = entry.get("person_ids") or []
    entity_id = entry.get("entity_id")
    entity_type = entry.get("entity_type")
    notes = []
    try:
        if org_ids:
            data = get("notes", params={"organization_id": org_ids[0]})
        elif entity_type == ENTITY_TYPE_OPPORTUNITY and entity_id:
            data = get("notes", params={"opportunity_id": entity_id})
        elif person_ids:
            data = get("notes", params={"person_id": person_ids[0]})
        elif entity_id:
            data = get("notes", params={"organization_id": entity_id})
        else:
            return []
        if isinstance(data, dict):
            notes = data.get("notes", [])
        else:
            notes = data if isinstance(data, list) else []
    except Exception:
        pass
    if not isinstance(notes, list):
        return []
    return notes[:15]


def enrich_company_like_entry(entry):
    """Resolve name/domain/org and notes via API (runs only on refresh)."""
    org_ids = entry.get("organization_ids", [])
    entity_id = entry.get("entity_id")
    entity_type = entry.get("entity_type")
    nested = entry.get("entity")
    nested = nested if isinstance(nested, dict) else {}

    org = {}
    opportunity = {}
    if org_ids:
        org = fetch_org(org_ids[0])

    if entity_type == ENTITY_TYPE_OPPORTUNITY and entity_id:
        opportunity = fetch_opportunity(entity_id)
        for oid in opportunity.get("organization_ids") or []:
            linked = fetch_org(oid)
            if not linked:
                continue
            if not org.get("domain") and linked.get("domain"):
                org["domain"] = linked["domain"]
            if not org.get("name") and linked.get("name"):
                org["name"] = linked["name"]
            if not org.get("id") and linked.get("id"):
                org["id"] = linked["id"]
            if org.get("domain") and org.get("name"):
                break

    if not org.get("name") and entity_id and entity_type != ENTITY_TYPE_OPPORTUNITY:
        org = fetch_org(entity_id)

    company_name = (
        nested.get("name")
        or org.get("name")
        or opportunity.get("name")
        or "Unknown"
    )
    domain = org.get("domain", "")

    notes = fetch_notes_for_entry(entry)

    return {
        "company_name": company_name,
        "domain": domain,
        "organization_id": str(org.get("id") or ""),
        "notes": notes,
    }


def enrich_contact_entry(entry):
    entity = entry.get("entity") or {}
    if not isinstance(entity, dict):
        entity = {}
    notes = fetch_notes_for_entry(entry)
    first = entity.get("first_name", "") or ""
    last = entity.get("last_name", "") or ""
    email = entity.get("primary_email", "") or ""
    name = f"{first} {last}".strip() or "Unknown Contact"
    return {
        "contact_name": name,
        "email": email,
        "first_name": first,
        "last_name": last,
        "notes": notes,
    }


def enrich_single_entry(entry: dict, list_config: dict) -> dict:
    is_contacts = list_config["id"] == 237644
    try:
        e = json.loads(json.dumps(entry))
        if is_contacts:
            e["_affinity"] = enrich_contact_entry(entry)
        else:
            e["_affinity"] = enrich_company_like_entry(entry)
        return e
    except Exception:
        return entry


def enrich_entries_for_cache(entries, list_config):
    out = []
    for entry in tqdm(entries, desc=f"Enrich {list_config['name'][:24]}"):
        out.append(enrich_single_entry(entry, list_config))
        time.sleep(0.05)
    return out


def fetch_list_entries(list_config, force_refresh=False):
    name = list_config["name"]
    list_id = list_config["id"]
    max_results = list_config.get("max")

    if not force_refresh and cache_is_fresh(name):
        return load_cache(name)

    print(f"Fetching {name} from API...")
    entries = get_all_pages(
        f"lists/{list_id}/list-entries",
        key="list_entries",
        max_results=max_results,
    )
    entries = enrich_entries_for_cache(entries, list_config)
    save_cache(name, entries)
    return entries


def build_company_doc(entry, list_name):
    aff = entry.get("_affinity") or {}
    if aff:
        company_name = aff.get("company_name", "Unknown")
        domain = aff.get("domain", "")
        org_id = aff.get("organization_id", "")
        notes = aff.get("notes") or []
    else:
        nested = entry.get("entity") or {}
        if not isinstance(nested, dict):
            nested = {}
        fn = nested.get("first_name", "") or ""
        ln = nested.get("last_name", "") or ""
        person_part = f"{fn} {ln}".strip()
        company_name = nested.get("name") or person_part or "Unknown"
        domain = ""
        org_id = ""
        notes = fetch_notes_for_entry(entry)

    created_at = entry.get("created_at", "")
    date_str = created_at[:10] if created_at else ""
    org_ids = entry.get("organization_ids", [])
    person_ids = entry.get("person_ids", [])

    note_texts = []
    for n in notes[:15]:
        content = (n.get("content") or "").strip()
        note_date = (n.get("created_at") or "")[:10]
        if content:
            note_texts.append(f"[{note_date}] {content}")

    text = f"""Company/Entity: {company_name}
Domain: {domain}
List: {list_name}
First tracked: {date_str}
Organization IDs: {org_ids}
Person IDs: {person_ids}

"""
    if note_texts:
        text += "Partner notes:\n" + "\n".join(note_texts)
    else:
        text += "No partner notes recorded."

    return Document(
        page_content=text,
        metadata={
            "company": company_name,
            "list": list_name,
            "date": date_str,
            "source": "affinity",
            "entity_id": str(entry.get("entity_id", "")),
            "domain": domain,
            "organization_id": org_id,
        },
    )


def build_contact_doc(entry, list_name):
    aff = entry.get("_affinity") or {}
    if aff:
        name = aff.get("contact_name", "Unknown Contact")
        email = aff.get("email", "")
        first = aff.get("first_name", "") or ""
        notes = aff.get("notes") or []
    else:
        entity = entry.get("entity") or {}
        if not isinstance(entity, dict):
            entity = {}
        first = entity.get("first_name", "") or ""
        last = entity.get("last_name", "") or ""
        email = entity.get("primary_email", "") or ""
        name = f"{first} {last}".strip() or "Unknown Contact"
        notes = fetch_notes_for_entry(entry)

    if not email and not first:
        return None

    created_at = entry.get("created_at", "")
    date_str = created_at[:10] if created_at else ""

    note_texts = [
        f"[{(n.get('created_at') or '')[:10]}] {(n.get('content') or '').strip()}"
        for n in notes
        if (n.get("content") or "").strip()
    ]

    text = f"""Contact: {name}
Email: {email}
List: {list_name}
First tracked: {date_str}

"""
    if note_texts:
        text += "Notes:\n" + "\n".join(note_texts)
    else:
        text += "No notes recorded."

    return Document(
        page_content=text,
        metadata={
            "company": name,
            "list": list_name,
            "date": date_str,
            "source": "affinity_contact",
            "entity_id": str(entry.get("entity_id", "")),
        },
    )


def main(force_refresh=False):
    placeholder_affinity = API_KEY in (None, "", "your-key-here")
    placeholder_openai = OPENAI_API_KEY in (None, "", "your-openai-key-here")
    if placeholder_affinity:
        raise SystemExit("Set a real AFFINITY_API_KEY in .env (not the placeholder).")
    if placeholder_openai:
        raise SystemExit("Set a real OPENAI_API_KEY in .env (not the placeholder).")

    all_docs = []

    for list_config in LISTS_CONFIG:
        print(f"\n{'=' * 50}")
        print(f"Processing: {list_config['name']}")
        entries = fetch_list_entries(list_config, force_refresh=force_refresh)
        print(f"Building documents from {len(entries)} entries...")

        is_contacts = list_config["id"] == 237644
        docs = []

        for entry in tqdm(entries, desc="Documents"):
            try:
                if is_contacts:
                    doc = build_contact_doc(entry, list_config["name"])
                else:
                    doc = build_company_doc(entry, list_config["name"])
                if doc:
                    docs.append(doc)
            except Exception:
                pass

        print(f"Built {len(docs)} documents from {list_config['name']}")
        all_docs.extend(docs)

    print(f"\nTotal documents: {len(all_docs)}")
    print("Building vector store...")

    if not all_docs:
        print("No documents to embed.")
        return []

    batch_size = 500
    first = all_docs[:batch_size]
    vectorstore = embed_with_retry(first, embeddings)
    if len(all_docs) > batch_size:
        for i in tqdm(range(batch_size, len(all_docs), batch_size)):
            batch = all_docs[i : i + batch_size]
            vectorstore.add_documents(batch)

    vectorstore.save_local("dvc_vectorstore")
    print("Vector store saved.")
    return all_docs


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    docs = main(force_refresh=force)
    print(f"\nDone. {len(docs)} total documents ingested.")
