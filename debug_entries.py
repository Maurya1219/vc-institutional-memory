import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AFFINITY_API_KEY")
AUTH = ("", API_KEY)

# Get first 3 entries with full detail
resp = requests.get(
    "https://api.affinity.co/lists/237676/list-entries",
    auth=AUTH,
    params={"page_size": 3},
)
data = resp.json()
print(json.dumps(data, indent=2))
