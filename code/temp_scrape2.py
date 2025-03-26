import requests
from datetime import datetime
import difflib
from IPython import embed

session = requests.Session()

def get_random_article():
    while True:
        res = session.get("https://en.wikipedia.org/w/api.php", params={
            "format": "json",
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": 1,
        })
        title = res.json()["query"]["random"][0]["title"]
        if title.startswith("List of") or "disambiguation" in title.lower():
            continue

        # Check if it's a disambiguation page
        page_info = session.get("https://en.wikipedia.org/w/api.php", params={
            "format": "json",
            "action": "query",
            "prop": "pageprops",
            "titles": title,
        }).json()
        page = next(iter(page_info["query"]["pages"].values()))
        if "pageprops" in page and "disambiguation" in page["pageprops"]:
            continue

        return title

def get_revision_as_of(title, date):
    res = session.get("https://en.wikipedia.org/w/api.php", params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": date,
        "rvprop": "content",
        "rvslots": "main"
    }).json()
    page = next(iter(res["query"]["pages"].values()))
    if "revisions" not in page:
        return None
    return page["revisions"][0]["slots"]["main"]["*"]

def get_latest_revision(title):
    res = session.get("https://en.wikipedia.org/w/api.php", params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvprop": "content",
        "rvslots": "main"
    }).json()
    page = next(iter(res["query"]["pages"].values()))
    return page["revisions"][0]["slots"]["main"]["*"]

def extract_lead(wikitext):
    embed()
    if not wikitext:
        return ""
    lines = wikitext.splitlines()
    lead = []
    for line in lines:
        if line.startswith("=="):
            break
        lead.append(line)
    return "\n".join(lead).strip()

def summary_stable_since_2016(title):
    old_rev = get_revision_as_of(title, "2016-12-31T23:59:59Z")
    new_rev = get_latest_revision(title)
    old_lead = extract_lead(old_rev)
    new_lead = extract_lead(new_rev)
    return old_lead == new_lead, old_lead, new_lead

# Example: find N articles with stable lead
stable_articles = []
attempts = 0
max_articles = 5

while len(stable_articles) < max_articles and attempts < 100:
    attempts += 1
    title = get_random_article()
    is_stable, old, new = summary_stable_since_2016(title)
    if is_stable:
        stable_articles.append((title, old))
        print(f"[{len(stable_articles)}] {title} — summary unchanged since 2016")


# Print summaries
print("\n=== Stable Summaries ===")
for title, summary in stable_articles:
    print(f"\n--- {title} ---\n{summary}\n")

embed()
"""
python3 -m code.temp_scrape2

"""
