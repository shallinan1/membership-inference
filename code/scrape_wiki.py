import requests
from datetime import datetime
import time
from langdetect import detect, DetectorFactory
from IPython import embed

# Ensure consistent results
DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        return f"Error detecting language: {e}"

def is_meaningful_title(title):
    lowered = title.lower()
    if lowered.startswith("timeline of") or lowered.startswith("list of"):
        return False
    if any(year in lowered for year in ["2024", "2025", "2026"]):
        return False
    return True

# Step 1: Fetch newly created pages in 2025 from recent changes API
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "list": "recentchanges",
    "rcstart": "2025-12-31T23:59:59Z",
    "rcend": "2025-01-01T00:00:00Z",
    "rcnamespace": 0,  # mainspace only
    "rctype": "new",
    "rcprop": "title|timestamp",
    "rclimit": "50",  # adjust up to 500 for bots
    "format": "json"
}

response = S.get(url=URL, params=PARAMS)
data = response.json()

# Filter titles
titles = [item['title'] for item in data['query']['recentchanges'] if is_meaningful_title(item['title'])]

# Step 2: Get content for each page
def get_page_extract(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "format": "json"
    }
    r = S.get(url=URL, params=params)
    result = r.json()
    page = next(iter(result['query']['pages'].values()))
    if "missing" in page:
        return 0
    extract = page.get('extract', '')
    if not extract.strip():
        return 0
    try:
        time.sleep(1)
        lang = detect_language(extract)
        if lang != "en":
            return 0
        return extract
    except:
        print("langdetect failed; returning raw extract.")
        return extract

# Example: Fetch and print
for title in titles[:500]:
    content = get_page_extract(title)
    if content == 0:
        continue
    if content.lstrip().startswith("=="):
        continue
    print(f"TITLE: {title}")
    print(content[:500])
    print("\n---\n")
    embed()

"""
python3 -m code.scrape_wiki
"""
