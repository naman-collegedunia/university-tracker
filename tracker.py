import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("university_monitor_data")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = DATA_DIR / "logs"

for directory in [DATA_DIR, SNAPSHOT_DIR, REPORT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is missing."
    )


# How many pages should be discovered from each university
MAX_PAGES_PER_UNIVERSITY = 150

# Maximum crawling depth from the starting URL
MAX_CRAWL_DEPTH = 3

# Request timeout
REQUEST_TIMEOUT = 30

# Delay between requests to the same university
REQUEST_DELAY = 1.0

# Only crawl pages that appear relevant to your website
RELEVANT_KEYWORDS = [
    "admission",
    "apply",
    "application",
    "entry",
    "requirements",
    "international",
    "india",
    "fees",
    "tuition",
    "cost",
    "scholarship",
    "funding",
    "deadline",
    "intake",
    "semester",
    "accommodation",
    "housing",
    "living",
    "visa",
    "immigration",
    "english",
    "ielts",
    "toefl",
    "course",
    "program",
    "programme",
    "undergraduate",
    "postgraduate",
    "master",
    "masters",
    "phd",
    "research",
    "deposit",
]


# ============================================================
# UNIVERSITIES
# ============================================================

UNIVERSITIES = {
    "University of Bristol": {
        "start_urls": [
            "https://www.bristol.ac.uk/international/countries/india.html",
        ],
        "allowed_domains": [
            "bristol.ac.uk",
        ],
    },

    "University of Leeds": {
        "start_urls": [
            "https://www.leeds.ac.uk/international-entry-requirements",
        ],
        "allowed_domains": [
            "leeds.ac.uk",
        ],
    },

    "University of York": {
        "start_urls": [
            "https://www.york.ac.uk/study/international/your-country/india/",
        ],
        "allowed_domains": [
            "york.ac.uk",
        ],
    },

    "University of Exeter": {
        "start_urls": [
            "https://www.exeter.ac.uk/study/international/yourcountry/india/",
        ],
        "allowed_domains": [
            "exeter.ac.uk",
        ],
    },

    "University of Warwick": {
        "start_urls": [
            "https://warwick.ac.uk/study/international/country/india/",
        ],
        "allowed_domains": [
            "warwick.ac.uk",
        ],
    },

    "UCL": {
        "start_urls": [
            "https://www.ucl.ac.uk/prospective-students/international/india",
        ],
        "allowed_domains": [
            "ucl.ac.uk",
        ],
    },

    "University of Malaya": {
        "start_urls": [
            "https://isc.um.edu.my/",
        ],
        "allowed_domains": [
            "um.edu.my",
        ],
    },

    "Freie Universität Berlin": {
        "start_urls": [
            "https://www.fu-berlin.de/en/studium/international/",
        ],
        "allowed_domains": [
            "fu-berlin.de",
        ],
    },

    "Ludwig Maximilian University Munich": {
        "start_urls": [
            "https://www.lmu.de/en/study/all-degrees-and-programs/international-full-time-students/",
        ],
        "allowed_domains": [
            "lmu.de",
        ],
    },

    "Technical University Berlin": {
        "start_urls": [
            "https://www.tu.berlin/en/studying/international-students/",
        ],
        "allowed_domains": [
            "tu.berlin",
        ],
    },
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_DIR / "crawler.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/151.0 Safari/537.36 "
        "UniversityInformationMonitor/1.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# HELPERS
# ============================================================

def safe_filename(name):
    """
    Convert university name into safe filename.
    """
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def normalize_url(url):
    """
    Remove fragments and normalize trailing slash.
    """
    parsed = urlparse(url)

    clean = parsed._replace(
        fragment="",
        query=parsed.query,
    )

    url = clean.geturl()

    if url.endswith("/") and parsed.path != "/":
        url = url[:-1]

    return url


def is_allowed_domain(url, allowed_domains):
    """
    Ensure URL belongs to official university domain.
    """

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return False

        hostname = hostname.lower()

        for domain in allowed_domains:
            domain = domain.lower()

            if (
                hostname == domain
                or hostname.endswith("." + domain)
            ):
                return True

        return False

    except Exception:
        return False


def looks_relevant(url):
    """
    Determine whether a discovered URL is relevant
    to the type of information we monitor.
    """

    url_lower = url.lower()

    return any(
        keyword in url_lower
        for keyword in RELEVANT_KEYWORDS
    )


def is_pdf(url):
    return url.lower().split("?")[0].endswith(".pdf")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Remove insignificant whitespace and normalize text
    so harmless formatting changes don't create false alerts.
    """

    text = text.replace("\xa0", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize repeated punctuation spacing
    text = re.sub(r"\s+([,:;.!?])", r"\1", text)

    return text.strip()


def extract_page_text(html):
    """
    Extract meaningful visible page content.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content elements
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "nav",
        "footer",
        "header",
        "form",
    ]):
        tag.decompose()

    # Remove common cookie/banner elements
    for tag in soup.find_all(
        ["div", "section"],
        class_=re.compile(
            r"(cookie|consent|banner|popup|modal)",
            re.I
        )
    ):
        tag.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True
    )

    lines = []

    for line in text.splitlines():

        line = normalize_text(line)

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# HASHING
# ============================================================

def content_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# FETCH PAGE
# ============================================================

def fetch_url(url):

    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        final_url = normalize_url(response.url)

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        return {
            "success": True,
            "url": final_url,
            "status_code": response.status_code,
            "content_type": content_type,
            "html": response.text,
        }

    except requests.RequestException as e:

        logger.warning(
            "Request failed: %s | %s",
            url,
            e,
        )

        return {
            "success": False,
            "url": url,
            "error": str(e),
        }


# ============================================================
# LINK DISCOVERY
# ============================================================

def discover_links(
    base_url,
    html,
    allowed_domains
):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    discovered = set()

    for link in soup.find_all("a", href=True):

        href = link.get("href")

        if not href:
            continue

        absolute_url = normalize_url(
            urljoin(base_url, href)
        )

        if not absolute_url.startswith(
            ("http://", "https://")
        ):
            continue

        if not is_allowed_domain(
            absolute_url,
            allowed_domains
        ):
            continue

        # Ignore obvious files that aren't useful
        if absolute_url.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".svg",
                ".webp",
                ".mp4",
                ".mp3",
                ".zip",
            )
        ):
            continue

        discovered.add(absolute_url)

    return discovered


# ============================================================
# CRAWLER
# ============================================================

def crawl_university(
    university_name,
    config
):

    allowed_domains = config["allowed_domains"]

    queue = []

    for url in config["start_urls"]:

        queue.append(
            (normalize_url(url), 0)
        )

    visited = set()

    pages = {}

    while queue and len(pages) < MAX_PAGES_PER_UNIVERSITY:

        url, depth = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        if depth > MAX_CRAWL_DEPTH:
            continue

        print(
            f"    Crawling [{len(pages)+1}] {url}"
        )

        result = fetch_url(url)

        if not result["success"]:
            continue

        final_url = result["url"]

        if not is_allowed_domain(
            final_url,
            allowed_domains
        ):
            continue

        content_type = result.get(
            "content_type",
            ""
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if "application/pdf" in content_type or is_pdf(final_url):

            pages[final_url] = {
                "url": final_url,
                "type": "pdf",
                "status_code": result["status_code"],
                "content": None,
                "hash": None,
            }

            continue

        # ----------------------------------------------------
        # HTML
        # ----------------------------------------------------

        html = result["html"]

        text = extract_page_text(html)

        if not text:
            continue

        # Only store relevant pages.
        # Starting URLs are always accepted.
        is_start_url = final_url in config["start_urls"]

        if is_start_url or looks_relevant(final_url):

            pages[final_url] = {
                "url": final_url,
                "type": "html",
                "status_code": result["status_code"],
                "content": text,
                "hash": content_hash(text),
            }

        # Discover additional official links
        links = discover_links(
            final_url,
            html,
            allowed_domains
        )

        for link in links:

            if link not in visited:

                queue.append(
                    (link, depth + 1)
                )

        time.sleep(REQUEST_DELAY)

    return pages


# ============================================================
# SNAPSHOT STORAGE
# ============================================================

def snapshot_path(university_name):

    return SNAPSHOT_DIR / (
        safe_filename(university_name)
        + ".json"
    )


def load_snapshot(university_name):

    path = snapshot_path(
        university_name
    )

    if not path.exists():
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        logger.error(
            "Unable to load snapshot %s: %s",
            university_name,
            e,
        )

        return {}


def save_snapshot(
    university_name,
    pages
):

    path = snapshot_path(
        university_name
    )

    snapshot = {
        "university": university_name,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "pages": pages,
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            snapshot,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# CHANGE DETECTION
# ============================================================

def compare_snapshots(
    old_snapshot,
    new_pages
):

    old_pages = old_snapshot.get(
        "pages",
        {}
    )

    changes = []

    # --------------------------------------------------------
    # New or modified pages
    # --------------------------------------------------------

    for url, new_page in new_pages.items():

        old_page = old_pages.get(url)

        if old_page is None:

            changes.append({
                "type": "NEW_PAGE",
                "url": url,
                "old_content": "",
                "new_content": new_page.get(
                    "content",
                    ""
                ),
            })

            continue

        old_hash = old_page.get(
            "hash"
        )

        new_hash = new_page.get(
            "hash"
        )

        if (
            old_hash
            and new_hash
            and old_hash != new_hash
        ):

            changes.append({
                "type": "CONTENT_CHANGED",
                "url": url,
                "old_content": old_page.get(
                    "content",
                    ""
                ),
                "new_content": new_page.get(
                    "content",
                    ""
                ),
            })

    # --------------------------------------------------------
    # Removed pages
    # --------------------------------------------------------

    for url in old_pages:

        if url not in new_pages:

            changes.append({
                "type": "PAGE_REMOVED",
                "url": url,
                "old_content": old_pages[url].get(
                    "content",
                    ""
                ),
                "new_content": "",
            })

    return changes


# ============================================================
# GEMINI CHANGE ANALYSIS
# ============================================================

def analyze_change(
    university_name,
    change
):

    prompt = f"""
You are monitoring the official website of:

{university_name}

IMPORTANT SOURCE RULE:
Only use the information supplied below.
Do NOT invent information.
Do NOT use knowledge from other websites.
Do NOT assume something changed unless the comparison supports it.

The website page is:

{change["url"]}

Change type:

{change["type"]}

OLD VERSION:
{change["old_content"]}

NEW VERSION:
{change["new_content"]}

Your task is to identify the REAL INFORMATION CHANGE.

Focus especially on information relevant to international students
and Indian students:

1. Tuition fees
2. Application fees
3. Admission requirements
4. Academic requirements
5. English-language requirements
6. Application deadlines
7. Intakes
8. Rolling admissions
9. Scholarships
10. Scholarship amounts
11. Scholarship deadlines
12. Deposits
13. Accommodation
14. Living costs
15. Visa / immigration guidance published by the university
16. Courses / programmes
17. Course duration
18. Application process
19. Required documents
20. Country-specific requirements
21. Important policy changes

IGNORE:
- Navigation changes
- Cookie notices
- Footer changes
- Menu changes
- Copyright text
- Social media links
- Pure formatting changes
- Tracking parameters
- Unimportant wording changes

Return STRICT JSON:

{{
  "real_change": true,
  "importance": "CRITICAL | IMPORTANT | MODERATE | LOW",
  "category": [
      "FEES",
      "DEADLINE",
      "ADMISSION",
      "SCHOLARSHIP",
      "ACCOMMODATION",
      "LIVING_COST",
      "VISA",
      "COURSE",
      "ENGLISH_REQUIREMENT",
      "APPLICATION_PROCESS",
      "OTHER"
  ],
  "summary": "Short description of what changed",
  "old_value": "Previous information",
  "new_value": "New information",
  "affected_students": "Who is affected",
  "effective_date": "If explicitly stated, otherwise null",
  "indian_student_impact": "Explain the practical impact for an Indian student",
  "confidence": "HIGH | MEDIUM | LOW"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    raw = response.text.strip()

    # Remove markdown JSON fences if Gemini adds them
    raw = re.sub(
        r"^```json\s*",
        "",
        raw,
        flags=re.I
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    try:

        return json.loads(raw)

    except json.JSONDecodeError:

        return {
            "real_change": False,
            "importance": "LOW",
            "category": ["OTHER"],
            "summary": "AI response could not be parsed.",
            "old_value": "",
            "new_value": "",
            "affected_students": "",
            "effective_date": None,
            "indian_student_impact": "",
            "confidence": "LOW",
            "raw_response": raw,
        }


# ============================================================
# REPORT GENERATION
# ============================================================

def create_daily_report(
    all_changes
):

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    report = {
        "date": today,
        "universities_checked": len(
            UNIVERSITIES
        ),
        "changes_detected": len(
            all_changes
        ),
        "changes": all_changes,
    }

    path = REPORT_DIR / (
        f"daily_report_{today}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


# ============================================================
# MAIN TRACKER
# ============================================================

def run_tracker():

    print("=" * 70)
    print("UNIVERSITY OFFICIAL WEBSITE MONITOR")
    print("=" * 70)

    all_changes = []

    for university_name, config in UNIVERSITIES.items():

        print()
        print("=" * 70)
        print(university_name)
        print("=" * 70)

        try:

            old_snapshot = load_snapshot(
                university_name
            )

            pages = crawl_university(
                university_name,
                config
            )

            print(
                f"Pages collected: {len(pages)}"
            )

            # ------------------------------------------------
            # First run
            # ------------------------------------------------

            if not old_snapshot:

                print(
                    "No previous snapshot. "
                    "Creating baseline."
                )

                save_snapshot(
                    university_name,
                    pages
                )

                continue

            # ------------------------------------------------
            # Compare
            # ------------------------------------------------

            changes = compare_snapshots(
                old_snapshot,
                pages
            )

            if not changes:

                print(
                    "No page changes detected."
                )

            else:

                print(
                    f"{len(changes)} raw changes detected."
                )

            # ------------------------------------------------
            # Analyze changes
            # ------------------------------------------------

            for change in changes:

                # Ignore page removal for now if
                # there is no meaningful content
                if (
                    change["type"]
                    == "PAGE_REMOVED"
                    and not change["old_content"]
                ):
                    continue

                print(
                    f"Analyzing: {change['url']}"
                )

                analysis = analyze_change(
                    university_name,
                    change
                )

                if analysis.get(
                    "real_change",
                    False
                ):

                    record = {
                        "university":
                            university_name,

                        "url":
                            change["url"],

                        "detected_at":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),

                        "change_type":
                            change["type"],

                        "analysis":
                            analysis,
                    }

                    all_changes.append(
                        record
                    )

                    print(
                        "  REAL CHANGE:"
                    )

                    print(
                        f"  {analysis.get('summary')}"
                    )

                    print(
                        f"  Importance: "
                        f"{analysis.get('importance')}"
                    )

                else:

                    print(
                        "  Ignored: "
                        "No meaningful student-facing change."
                    )

            # ------------------------------------------------
            # Save new snapshot
            # ------------------------------------------------

            save_snapshot(
                university_name,
                pages
            )

        except Exception as e:

            logger.exception(
                "University failed: %s",
                university_name
            )

            print(
                f"ERROR: {e}"
            )

    # --------------------------------------------------------
    # Daily report
    # --------------------------------------------------------

    report_path = create_daily_report(
        all_changes
    )

    print()
    print("=" * 70)
    print("CRAWL COMPLETE")
    print("=" * 70)

    print(
        f"Meaningful changes: {len(all_changes)}"
    )

    print(
        f"Report: {report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_tracker()
