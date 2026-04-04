import json
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("AFFINITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_URL = "https://api.affinity.co"
AUTH = ("", API_KEY)

MASTER_LIST_ID = 237676

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


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
    resp = requests.get(f"{BASE_URL}/{endpoint}", auth=AUTH, params=params)
    resp.raise_for_status()
    return resp.json()


def get_all_pages(endpoint, key, params=None):
    results = []
    page_token = None
    while True:
        p = params.copy() if params else {}
        p["page_size"] = 100
        if page_token:
            p["page_token"] = page_token
        data = get(endpoint, params=p)
        items = data.get(key, data) if isinstance(data, dict) else data
        if isinstance(items, list):
            results.extend(items)
        next_token = data.get("next_page_token") if isinstance(data, dict) else None
        if not next_token:
            break
        page_token = next_token
        time.sleep(0.2)
    return results


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


def fetch_notes_for_entity(entity_id):
    try:
        notes = get("notes", params={"organization_id": entity_id})
        if isinstance(notes, dict):
            return notes.get("notes", [])
        return notes
    except Exception:
        return []


def fetch_notes_for_opportunity(opportunity_id):
    try:
        notes = get("notes", params={"opportunity_id": opportunity_id})
        if isinstance(notes, dict):
            return notes.get("notes", [])
        return notes
    except Exception:
        return []


# Affinity list type "opportunity" uses entity_type 8; entity_id is an opportunity, not an org.
ENTITY_TYPE_OPPORTUNITY = 8


def build_document(entry, company_name, domain, linked_org_id, notes):
    created_at = entry.get("created_at", "")

    date_str = ""
    if created_at:
        try:
            date_str = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            ).strftime("%B %Y")
        except Exception:
            date_str = created_at[:10]

    note_texts = []
    for n in notes[:10]:
        content = n.get("content", "")
        note_date = n.get("created_at", "")[:10]
        if content:
            note_texts.append(f"[{note_date}] {content}")

    text = f"""Company: {company_name}
Domain: {domain}
First tracked: {date_str}
List: Master DVC Advantage Tracker

"""
    if note_texts:
        text += "Partner notes:\n" + "\n".join(note_texts)
    else:
        text += "No partner notes recorded."

    metadata = {
        "company": company_name,
        "domain": domain,
        "entity_id": str(entry.get("entity_id", "")),
        "organization_id": str(linked_org_id or ""),
        "source": "affinity",
        "list": "Master DVC Advantage Tracker",
        "date": created_at[:10] if created_at else "",
    }

    return Document(page_content=text, metadata=metadata)


def main():
    placeholder_affinity = API_KEY in (None, "", "your-key-here")
    placeholder_openai = OPENAI_API_KEY in (None, "", "your-openai-key-here")
    if placeholder_affinity:
        raise SystemExit("Set a real AFFINITY_API_KEY in .env (not the placeholder).")
    if placeholder_openai:
        raise SystemExit("Set a real OPENAI_API_KEY in .env (not the placeholder).")

    print("Fetching Master DVC Advantage Tracker entries...")
    entries = get_all_pages(
        f"lists/{MASTER_LIST_ID}/list-entries",
        key="list_entries",
    )
    print(f"Found {len(entries)} entries")

    docs = []
    print("Enriching with org data and notes...")
    for entry in tqdm(entries):
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

        if entity_type == ENTITY_TYPE_OPPORTUNITY and entity_id:
            notes = fetch_notes_for_opportunity(entity_id)
        else:
            notes = fetch_notes_for_entity(org.get("id") or entity_id)

        doc = build_document(
            entry,
            company_name,
            domain,
            org.get("id"),
            notes,
        )
        docs.append(doc)
        time.sleep(0.1)

    print(f"\nBuilding vector store from {len(docs)} documents...")
    vectorstore = embed_with_retry(docs, embeddings)
    vectorstore.save_local("dvc_vectorstore")
    print("Vector store saved to dvc_vectorstore/")

    print("\nSample — first 3 documents:")
    for doc in docs[:3]:
        print(f"\n{doc.metadata['company']}")
        print(doc.page_content[:300])
        print("---")

    return docs, vectorstore


if __name__ == "__main__":
    docs, vs = main()
    print(f"\nDone. {len(docs)} companies ingested into vector store.")
