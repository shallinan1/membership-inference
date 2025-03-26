import requests
from datetime import datetime
import time
from langdetect import detect, DetectorFactory
from IPython import embed

# Ensure consistent results
DetectorFactory.seed = 0

def detect_language(text):
    try:
        language = detect(text)
        return language
    except Exception as e:
        return f"Error detecting language: {e}"

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

titles = [item['title'] for item in data['query']['recentchanges']]

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
    # Check if page exists
    if "missing" in page:
        return "Page does not exist."    

    extract = page.get('extract', '')
    if not extract.strip():
        return 0 #"No extract available for this page."

    try:
        time.sleep(1)
        t = detect_language(extract)
        if t != "en":
            return 0 #"Article not in English, it is in: " + t
        else:
            # print("Article was in English! Here it is..")
            return extract
    except:
        print("langdetect stopped working, so just returning the extract found without checkingthe language!")
        return extract

# Example: Fetch and print first 5
for title in titles[:500]:
    content = get_page_extract(title)
    if content == 0:
        continue
    if content.lstrip()[0].startswith("=="):
        continue
    print(f"TITLE: {title}")
    print(content[:500])  # print first 500 characters
    print("\n---\n")
    embed()

"""
python3 -m code.scrape_wiki
"""