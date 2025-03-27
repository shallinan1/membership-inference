import requests
import difflib
import time
from IPython import embed
from wikitextparser import remove_markup, parse
import re
from tqdm import tqdm
import Levenshtein
from datetime import datetime
from unidecode import unidecode
from code.utils import save_to_jsonl

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

def remove_urls(text):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def get_random_article():
    while True:
        res = session.get(BASE_URL, params={
            "format": "json",
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": 1,
        })
        title = res.json()["query"]["random"][0]["title"]
        if (
            title.startswith("List of") or
            "disambiguation" in title.lower() or
            is_stub(title)
        ):
            print(f"[SKIPPED] {title} — list/disambig/stub")
            continue


        # Check disambiguation via pageprops
        page_info = session.get(BASE_URL, params={
            "format": "json",
            "action": "query",
            "prop": "pageprops",
            "titles": title,
        }).json()
        page = next(iter(page_info["query"]["pages"].values()))
        if "pageprops" in page and "disambiguation" in page["pageprops"]:
            continue

        return title

def get_revision_id_as_of(title, date="2016-12-31T23:59:59Z"):
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": date,
        "rvprop": "ids",
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    if "revisions" not in page:
        return None
    return page["revisions"][0]["revid"]

def get_revision_wikitext(revid):
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "revids": revid,
        "rvprop": "content",
        "rvslots": "main"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    try:
        output = page["revisions"][0]["slots"]["main"]["*"]
        return output
    except:
        return None

def get_latest_revision(title):
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvprop": "content|timestamp",
        "rvslots": "main"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    revision = page["revisions"][0]
    wikitext = revision["slots"]["main"]["*"]
    timestamp = revision["timestamp"]
    return wikitext, timestamp

def extract_plain_summary(wikitext):
    try:
        parsed = parse(wikitext)
    except:
        return ""

    parsed = parse(wikitext)

    summary = parsed.sections[0].plain_text().strip()
    summary = re.sub(r' {2,}', ' ', summary)

    if summary.startswith("thumb"):
        index = summary.find("\n")
        if index != -1:
            summary = summary[index+1:].strip()
    if "\n\n\nthumb" in summary:
        index = summary.find("\n\n\nthumb")
        if index != 1:
            summary = summary[:index].split()

    if summary == "":
        return ""
    
    summary = remove_urls(summary).strip()
    summary = re.sub(r'\(\s*[\.,;:\-!?]*\s*\)', '', summary) # remove empty parens

    if summary.startswith("is a"):
        return ""
    
    if len(summary.split()) < 5:
        return ""   

    if "accessed on" in summary:
        return "" # Parsing error

    return unidecode(summary)

results = []

raw_date = "2016-12-31T23:59:59Z"
parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

def compare_summaries(title):
    revid = get_revision_id_as_of(title)
    
    if revid is None:
        return False
    
    old_wikitext = get_revision_wikitext(revid)
    if old_wikitext is None:
        return False
    new_wikitext, timestamp = get_latest_revision(title)

    # Ensure the latest revision is from Jan 2024 or later
    latest_revision_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if latest_revision_date < datetime(2024, 1, 1, tzinfo=latest_revision_date.tzinfo):
        print(f"[SKIPPED: TOO OLD] {title} — Last edit on {latest_revision_date.date()}")
        return False

    old_summary = extract_plain_summary(old_wikitext)
    new_summary = extract_plain_summary(new_wikitext)

    if old_summary == "" or new_summary == "":
        return False

    if old_summary == new_summary:
        print(f"[UNCHANGED] {title}")
        return False
    else:
        diff_chars = Levenshtein.distance(old_summary, new_summary)

        print(f"[CHANGED] {title} — Levenshtein Distance: {diff_chars} characters")

        results.append({
            "title": title,
            "old_summary": old_summary,
            "new_summary": new_summary,
            "char_difference": diff_chars,
            "percent_diff": (2*diff_chars)/(len(old_summary) + len(new_summary)),
            "first_retrieved_date": parsed_date.date().isoformat(),
            "last_edit_date": latest_revision_date.date().isoformat()
        })

# Run on N random articles
count = 0
tries = 0
max_tries = 500

for tries in tqdm(range(max_tries)):
    tries += 1
    title = get_random_article()
    result = compare_summaries(title)
    if result:
        count += 1
    time.sleep(0.1)

save_to_jsonl(results, "data/wikiMIA_hard/scraped/scraped.jsonl")
embed()
"""
python3 -m data.wikiMIA_hard.scrape_articles
"""
