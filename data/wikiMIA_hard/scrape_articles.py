# Wikipedia article scraper for membership inference attack (MIA) research
# This script compares Wikipedia article summaries from 2016 vs current versions
# to identify articles that have changed significantly over time

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
import json

# Create a persistent session for API requests to Wikipedia
session = requests.Session()
BASE_URL = "https://en.wikipedia.org/w/api.php"

def is_stub(title):
    """
    Check if a Wikipedia article is a stub (short, incomplete article)
    by examining its categories for any containing 'stubs'
    """
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
    """Remove URLs from text to clean up article summaries"""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def get_random_article():
    """
    Get a random Wikipedia article title, filtering out unwanted types:
    - List articles (e.g., "List of countries")
    - Disambiguation pages
    - Stub articles (short, incomplete articles)
    
    Continues searching until it finds a suitable main namespace article
    """
    while True:
        # Get a random article from main namespace (0)
        res = session.get(BASE_URL, params={
            "format": "json",
            "action": "query",
            "list": "random",
            "rnnamespace": 0,  # Main namespace only
            "rnlimit": 1,
        })
        title = res.json()["query"]["random"][0]["title"]
        
        # Skip unwanted article types
        if (
            title.startswith("List of") or
            "disambiguation" in title.lower() or
            is_stub(title)
        ):
            print(f"[SKIPPED] {title} — list/disambig/stub")
            continue

        # Double-check for disambiguation pages via pageprops
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
    """
    Get the revision ID of an article as it existed on a specific date
    Default date is end of 2016 for the membership inference study
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",  # Get revisions going backwards in time
        "rvstart": date,   # Start from this date and go backwards
        "rvprop": "ids",   # Only need the revision ID
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    if "revisions" not in page:
        return None
    return page["revisions"][0]["revid"]

def get_revision_wikitext(revid):
    """
    Get the raw wikitext content for a specific revision ID
    Returns the markup text that Wikipedia uses internally
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "revids": revid,
        "rvprop": "content",
        "rvslots": "main"  # Get content from main slot
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    try:
        output = page["revisions"][0]["slots"]["main"]["*"]
        return output
    except:
        return None

def get_latest_revision(title):
    """
    Get the current version of an article's wikitext and its timestamp
    Returns both the content and when it was last modified
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,  # Only get the most recent revision
        "rvprop": "content|timestamp",  # Get both content and edit timestamp
        "rvslots": "main"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    revision = page["revisions"][0]
    wikitext = revision["slots"]["main"]["*"]
    timestamp = revision["timestamp"]
    return wikitext, timestamp

def extract_plain_summary(wikitext):
    """
    Convert Wikipedia wikitext markup to plain text summary
    Extracts only the first section (intro/summary) and cleans it up
    
    Filters out articles with:
    - Empty or very short summaries (< 5 words)
    - Summaries that start with "is a" (usually parsing errors)
    - Parsing artifacts like "accessed on" or "thumb" remnants
    """
    try:
        parsed = parse(wikitext)
    except:
        return ""

    parsed = parse(wikitext)

    # Extract first section (intro/summary) as plain text
    summary = parsed.sections[0].plain_text().strip()
    summary = re.sub(r' {2,}', ' ', summary)  # Collapse multiple spaces

    # Clean up common parsing artifacts from images/thumbnails
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
    
    # Remove URLs and empty parentheses
    summary = remove_urls(summary).strip()
    summary = re.sub(r'\(\s*[\.,;:\-!?]*\s*\)', '', summary)

    # Filter out low-quality summaries
    if summary.startswith("is a"):
        return ""  # Usually indicates parsing error
    
    if len(summary.split()) < 5:
        return ""  # Too short to be meaningful

    if "accessed on" in summary:
        return ""  # Parsing error artifact

    # Convert Unicode characters to ASCII equivalents
    return unidecode(summary)

# Global variables for the scraping process
results = []  # Store results (currently unused, writing directly to file instead)

# Target date for historical comparison (end of 2016)
raw_date = "2016-12-31T23:59:59Z"
parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

def compare_summaries(title):
    """
    Compare Wikipedia article summaries from 2016 vs current version
    
    This is the core function that:
    1. Gets the 2016 version of the article
    2. Gets the current version 
    3. Extracts and compares the summaries
    4. Calculates similarity metrics
    5. Filters for articles with recent edits (2024+)
    
    Returns article data if changed significantly, False otherwise
    """
    # Get the revision ID from 2016
    revid = get_revision_id_as_of(title)
    
    if revid is None:
        return False  # Article didn't exist in 2016
    
    # Get the old wikitext content
    old_wikitext = get_revision_wikitext(revid)
    if old_wikitext is None:
        return False  # Couldn't retrieve old content
        
    # Get current version
    new_wikitext, timestamp = get_latest_revision(title)

    # Only include articles that have been edited recently (2024+)
    # This ensures we're looking at articles that are actively maintained
    latest_revision_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if latest_revision_date < datetime(2024, 1, 1, tzinfo=latest_revision_date.tzinfo):
        print(f"[SKIPPED: TOO OLD] {title} — Last edit on {latest_revision_date.date()}")
        return False

    # Extract clean summaries from both versions
    old_summary = extract_plain_summary(old_wikitext)
    new_summary = extract_plain_summary(new_wikitext)

    # Skip if either summary is empty/invalid
    if old_summary == "" or new_summary == "":
        return False

    # Check if the summaries have changed
    if old_summary == new_summary:
        print(f"[UNCHANGED] {title}")
        return False
    else:
        # Calculate edit distance between summaries
        diff_chars = Levenshtein.distance(old_summary, new_summary)

        print(f"[CHANGED] {title} — Levenshtein Distance: {diff_chars} characters")

        # Return structured data about the changes
        return {
            "title": title,
            "old_summary": old_summary,
            "new_summary": new_summary,
            "char_difference": diff_chars,
            "percent_diff": (2*diff_chars)/(len(old_summary) + len(new_summary)),
            "first_retrieved_date": parsed_date.date().isoformat(),
            "last_edit_date": latest_revision_date.date().isoformat()
        }

# Main scraping loop
# This searches through random Wikipedia articles looking for ones that have
# changed significantly between 2016 and now (with recent 2024+ edits)

count = 0  # Number of articles found with significant changes
tries = 0  # Total articles examined
max_tries = 200000  # Maximum articles to examine

print(f"Starting Wikipedia article scraping for membership inference dataset...")
print(f"Looking for articles changed between 2016 and present (with 2024+ edits)")
print(f"Will examine up to {max_tries} random articles")

for tries in tqdm(range(max_tries)):
    tries += 1
    title = get_random_article()  # Get a random suitable article
    result = compare_summaries(title)  # Compare 2016 vs current summary
    
    if result:  # If the article has changed significantly
        count += 1
        # Save the result immediately to avoid losing data
        with open("data/wikiMIA_hard/scraped/scraped.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    
    # Rate limiting - be nice to Wikipedia's servers
    time.sleep(0.1)

print(f"Scraping complete. Found {count} changed articles out of {tries} examined.")

# Backup save (though results list is currently unused)
save_to_jsonl(results, "data/wikiMIA_hard/scraped/scraped_temp_copy.jsonl")

# Drop into interactive mode for debugging/analysis
embed()

"""
To run this script:
python3 -m data.wikiMIA_hard.scrape_articles
"""
