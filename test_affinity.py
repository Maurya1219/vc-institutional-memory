import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AFFINITY_API_KEY", "")


def test(endpoint):
    resp = requests.get(
        f"https://api.affinity.co/{endpoint}",
        auth=("", API_KEY),
    )
    print(f"\n--- {endpoint} ---")
    print(f"status: {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2)[:2000])
    except requests.exceptions.JSONDecodeError:
        print(resp.text[:2000])


if __name__ == "__main__":
    if not API_KEY:
        print("Set AFFINITY_API_KEY in .env")
    else:
        test("lists")
        test("persons?page_size=3")
        test("organizations?page_size=3")
        test("opportunities?page_size=3")
