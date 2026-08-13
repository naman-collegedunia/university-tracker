import os
import json
import requests
from bs4 import BeautifulSoup
from google import genai

# Target university pages (India specific entry requirement / admission portals)
TARGET_URLS = {
    "University of Bristol": "https://www.bristol.ac.uk/international/countries/india.html",
    "University of Leeds": "https://www.leeds.ac.uk/international-entry-requirements",
    "University of York": "https://www.york.ac.uk/study/international/your-country/india/",
    "University of Exeter": "https://www.exeter.ac.uk/study/international/yourcountry/india/",
    "University of Warwick": "https://warwick.ac.uk/study/international/country/india/",
    "UCL University": "https://www.ucl.ac.uk/prospective-students/international/india",
    "University of Malaya": "https://isc.um.edu.my/",
    "Free University of Berlin": "https://www.fu-berlin.de/en/studium/international/",
    "Ludwig Maximilian University Munich": "https://www.lmu.de/en/study/all-degrees-and-programs/international-full-time-students/",
    "Technical University Berlin": "https://www.tu.berlin/en/studying/international-students/"
}

SNAPSHOT_FILE = "snapshots.json"

def get_page_text(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers, timeout=20)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove clutter like header scripts and styling
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()
        
    text = soup.get_text(separator=' ', strip=True)
    return text[:8000] # Capture visible content snippet

def summarize_update(uni_name, url, old_text, new_text):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("API key missing.")
        return
        
    client = genai.Client(api_key=api_key)
    prompt = f"""
    The official website for {uni_name} ({url}) was updated.
    
    Identify what changed between the old text and new text. Focus specifically on:
    - Deadlines, entry requirements, fees (converted roughly to INR), scholarships, or visa rules for Indian students.
    
    Write a short update post (2-3 paragraphs) formatted for an Indian student news blog.
    
    OLD TEXT:
    {old_text[:2000]}
    
    NEW TEXT:
    {new_text[:2000]}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    output_filename = f"updates_{uni_name.lower().replace(' ', '_')}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved update report to {output_filename}")

def run_tracker():
    old_snapshots = {}
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            old_snapshots = json.load(f)

    new_snapshots = {}

    for uni_name, url in TARGET_URLS.items():
        print(f"Checking: {uni_name}...")
        try:
            current_text = get_page_text(url)
            new_snapshots[uni_name] = current_text
            
            if uni_name in old_snapshots:
                if old_snapshots[uni_name] != current_text:
                    print(f"CHANGE DETECTED for {uni_name}!")
                    summarize_update(uni_name, url, old_snapshots[uni_name], current_text)
                else:
                    print(f"No changes for {uni_name}.")
            else:
                print(f"Initial snapshot stored for {uni_name}.")
        except Exception as e:
            print(f"Failed to check {uni_name}: {e}")

    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_snapshots, f, indent=2)

if __name__ == "__main__":
    run_tracker()
