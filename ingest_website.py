import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://www.dallasvc.com"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def get_all_links(base_url: str) -> list[str]:
    try:
        resp = requests.get(base_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = set()
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc == urlparse(base_url).netloc:
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.add(clean)
        return list(links)
    except Exception as e:
        print(f"Failed to get links: {e}")
        return [base_url]


def scrape_page(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""


def ingest_website():
    print(f"Discovering pages on {BASE_URL}...")
    urls = get_all_links(BASE_URL)
    urls = list(set([BASE_URL] + urls))
    print(f"Found {len(urls)} pages")

    all_docs = []
    for url in urls:
        print(f"Scraping {url}...")
        text = scrape_page(url)
        if not text or len(text) < 100:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": "website",
                        "url": url,
                        "site": "dallasvc.com",
                        "chunk": i,
                    },
                )
            )
        time.sleep(0.5)

    if not all_docs:
        print("No content found.")
        return []

    print(f"\n{len(all_docs)} chunks scraped. Merging into vector store...")

    if os.path.exists("dvc_vectorstore/index.faiss"):
        vectorstore = FAISS.load_local(
            "dvc_vectorstore",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(all_docs)
    else:
        vectorstore = FAISS.from_documents(all_docs, embeddings)

    vectorstore.save_local("dvc_vectorstore")
    print(f"Done. {len(all_docs)} website chunks added.")
    return all_docs


if __name__ == "__main__":
    ingest_website()
