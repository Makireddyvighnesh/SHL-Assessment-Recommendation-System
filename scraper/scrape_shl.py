"""
Robust SHL Individual Test Solutions Scraper
Updated: Structured detail extraction
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE = 12
MIN_EXPECTED = 377

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------

def get_soup(url: str, params: dict = None) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


# -----------------------------------------------------------
# Parse Listing Page
# -----------------------------------------------------------

def parse_listing_page(soup: BeautifulSoup):
    assessments = []

    tables = soup.find_all("table")
    if not tables:
        return []

    target_table = None
    for table in tables:
        if "Individual Test Solutions" in table.get_text():
            target_table = table
            break

    if target_table is None:
        target_table = max(tables, key=lambda t: len(t.find_all("tr")))

    rows = target_table.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue

        link = cells[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        url = urljoin(BASE_URL, href)

        remote_support = "Yes" if len(cells) > 1 and cells[1].find("img") else "No"
        adaptive_support = "Yes" if len(cells) > 2 and cells[2].find("img") else "No"

        test_types = []
        if len(cells) > 3:
            raw = cells[3].get_text(" ", strip=True)
            for ch in raw.split():
                if ch in TEST_TYPE_MAP:
                    mapped = TEST_TYPE_MAP[ch]
                    if mapped not in test_types:
                        test_types.append(mapped)

        assessments.append({
            "name": name,
            "url": url,
            "remote_support": remote_support,
            "adaptive_support": adaptive_support,
            "test_type": test_types,
            "description": "",
            "duration": None,
            "job_levels": [],
            "languages": []
        })

    return assessments


# -----------------------------------------------------------
# Structured Detail Scraper
# -----------------------------------------------------------

def scrape_detail(url: str):
    try:
        soup = get_soup(url)

        # Remove layout noise
        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()

        data = {
            "description": "",
            "duration": None,
            "job_levels": [],
            "languages": []
        }

        # Loop through all h4 sections
        for header in soup.find_all("h4"):
            title = header.get_text(strip=True).lower()

            value_tag = header.find_next_sibling("p")
            if not value_tag:
                continue

            text = value_tag.get_text(" ", strip=True)

            # ---- DESCRIPTION ----
            if "description" in title:
                data["description"] = text

            # ---- JOB LEVELS ----
            elif "job levels" in title:
                data["job_levels"] = [
                    j.strip() for j in text.split(",") if j.strip()
                ]

            # ---- LANGUAGES ----
            elif "languages" in title:
                data["languages"] = [
                    l.strip() for l in text.split(",") if l.strip()
                ]

            # ---- ASSESSMENT LENGTH ----
            elif "assessment length" in title:
                # Pattern: = 30
                match = re.search(r"=\s*(\d+)", text)
                if match:
                    data["duration"] = int(match.group(1))
                else:
                    # fallback: first number
                    match2 = re.search(r"(\d+)", text)
                    if match2:
                        data["duration"] = int(match2.group(1))
                    else:
                        data["duration"] = text  # keep raw string if needed

        return data

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {
            "description": "",
            "duration": None,
            "job_levels": [],
            "languages": []
        }
    
# -----------------------------------------------------------
# Scrape All Listings
# -----------------------------------------------------------

def scrape_all():
    all_assessments = []
    seen_urls = set()
    page_index = 0

    while True:
        start = page_index * PAGE_SIZE
        print(f"Scraping page {page_index+1} (start={start})")

        soup = get_soup(CATALOG_URL, params={"start": start, "type": 1})
        items = parse_listing_page(soup)

        if not items:
            print("No more items found.")
            break

        new_count = 0
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_assessments.append(item)
                new_count += 1

        print(f"  Found {len(items)} items, {new_count} new")

        if new_count == 0:
            break

        page_index += 1
        time.sleep(0.8)

    print(f"\nTotal scraped: {len(all_assessments)}")

    if len(all_assessments) < MIN_EXPECTED:
        raise ValueError(
            f"Incomplete scrape: {len(all_assessments)} < {MIN_EXPECTED}"
        )

    return all_assessments


# -----------------------------------------------------------
# Enrich Details
# -----------------------------------------------------------

def enrich_details(data):
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{len(data)}] {item['name']}")

        details = scrape_detail(item["url"])
        print(details)

        item["description"] = details["description"]
        item["duration"] = details["duration"]
        item["job_levels"] = details["job_levels"]
        item["languages"] = details["languages"]

        time.sleep(0.5)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":

    print("="*60)
    print("SHL Individual Test Solutions Scraper (Updated)")
    print("="*60)

    assessments = scrape_all()
    enrich_details(assessments)

    with open("shl_assessments.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print("\nSaved to shl_assessments.json")
    print(f"Total: {len(assessments)}")
    print(f"With description: {sum(1 for a in assessments if a['description'])}")
    print(f"With duration: {sum(1 for a in assessments if a['duration'])}")
    print(f"With job_levels: {sum(1 for a in assessments if a['job_levels'])}")
    print(f"With languages: {sum(1 for a in assessments if a['languages'])}")