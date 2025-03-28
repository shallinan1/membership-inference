import requests
import re
import time
from datetime import datetime
from tqdm import tqdm
from unidecode import unidecode
import json
from datetime import timezone
from IPython import embed

session = requests.Session()
BASE_URL = "https://en.wikipedia.org/w/api.php"

def is_stub(title):
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "categories",
        "titles": title,
        "cllimit": "max"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    if "categories" not in page:
        return False
    return any("stubs" in cat["title"].lower() for cat in page["categories"])

def get_random_article_with_metadata():
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 1,
        "prop": "extracts|revisions",
        "explaintext": 1,
        "exintro": 1,
        "rvprop": "timestamp",
    })
    
    page = next(iter(res.json()["query"]["pages"].values()))
    title = page.get("title")
    extract = page.get("extract", "").strip()
    extract = unidecode(extract)

    try:
        last_edit = datetime.fromisoformat(page["revisions"][0]["timestamp"].replace("Z", "+00:00"))
    except:
        return None

    return {
        "title": title,
        "summary": extract,
        "last_edit": last_edit
    }

def get_creation_date(title):
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "newer",
        "rvprop": "timestamp"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    try:
        return datetime.fromisoformat(page["revisions"][0]["timestamp"].replace("Z", "+00:00"))
    except:
        return None

# Collecting articles
max_count = 500000

old_article_count=0
new_article_count=0
pbar = tqdm(range(max_count))
for m in pbar:
    pbar.set_postfix({'old': old_article_count, 'new':  new_article_count})
    result = get_random_article_with_metadata()

    if result is None:
        continue

    title = result["title"]
    summary = result["summary"]

    # Filtering
    if (
        title.startswith("List of") or
        "disambiguation" in title.lower() or
        is_stub(title) or
        summary == "" or
        len(summary.split()) < 5
    ):
        continue

    if "may refer to" in result["summary"]:
        continue

    # Case 1: old articles (last edited before 2017)
    if result["last_edit"] < datetime(2017, 1, 1, tzinfo=timezone.utc):
        result["created"] = get_creation_date(title).isoformat()
        result["label"] = 0
        result["last_edit"] = result["last_edit"].isoformat()
        old_article_count+=1
        with open("data/wikiMIA_2024_plus/scraped/scraped_5.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        continue

    # Case 2: new articles (created in 2024+)
    created = get_creation_date(title)
    if created and created >= datetime(2024, 1, 1, tzinfo=timezone.utc):
        result["created"] = created.isoformat()
        result["label"] = 1
        result["last_edit"] = result["last_edit"].isoformat()
        new_article_count+=1
        with open("data/wikiMIA_2024_plus/scraped/scraped_5.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    time.sleep(0.01)

embed()

# save_to_jsonl(results, "data/wikiMIA_hard/scraped/scraped_temp_copy.jsonl")
# embed()
"""
python3 -m data.wikiMIA_2024_plus.scrape_articles
"""
