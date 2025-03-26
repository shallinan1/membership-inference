import requests
import difflib
import time
from IPython import embed
from wikitextparser import remove_markup, parse
import re
from tqdm import tqdm

session = requests.Session()
BASE_URL = "https://en.wikipedia.org/w/api.php"

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
        if title.startswith("List of") or "disambiguation" in title.lower():
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
    return page["revisions"][0]["slots"]["main"]["*"]

def get_latest_revision(title):
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvprop": "content",
        "rvslots": "main"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    return page["revisions"][0]["slots"]["main"]["*"]

def extract_plain_summary(wikitext):
    parsed = parse(wikitext)
    summary = parsed.sections[0].plain_text().strip()
    summary = re.sub(r' {2,}', ' ', summary)

    summary = remove_urls(summary).strip()
    if summary.startswith("is a"):
        embed()


    return summary

def compare_summaries(title):
    revid = get_revision_id_as_of(title)
    
    if revid is None:
        return None
    old_wikitext = get_revision_wikitext(revid)
    new_wikitext = get_latest_revision(title)
    old_summary = extract_plain_summary(old_wikitext)
    new_summary = extract_plain_summary(new_wikitext)

    if old_summary == new_summary:
        print(f"[UNCHANGED] {title}")
    else:
        print(f"\n[CHANGED] {title}")
        print(">>> DIFF:")
        diff = difflib.unified_diff(
            old_summary.splitlines(),
            new_summary.splitlines(),
            fromfile="2016",
            tofile="current",
            lineterm=""
        )
        for line in diff:
            print(line)
    return True

# Run on N random articles
n_articles = 100
count = 0
tries = 0
max_tries = 100

for tries in tqdm(range(max_tries)):
    tries += 1
    title = get_random_article()
    result = compare_summaries(title)
    if result:
        count += 1
    time.sleep(1)

embed()
"""
python3 -m code.temp_scrape2

"""
