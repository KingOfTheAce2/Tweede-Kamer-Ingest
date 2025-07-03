
import sqlite3
import requests
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi
import os
from typing import List, Dict, Optional # Import Optional


DB_PATH = "progress.sqlite3"
GLOBAL_BATCH_LIMIT = 1000 # Set a global limit for total documents to fetch in one run


def get_skiptoken(category: str) -> int:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS progress(category TEXT PRIMARY KEY, skiptoken INTEGER)"
    )
    cur = con.execute("SELECT skiptoken FROM progress WHERE category=?", (category,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else -1


def save_skiptoken(category: str, skiptoken: int) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "REPLACE INTO progress(category, skiptoken) VALUES(?,?)", (category, skiptoken)
    )
    con.commit()
    con.close()


# Renamed fetch_batch to fetch_all_docs to reflect its new looping behavior
def fetch_all_docs(category: str) -> List[Dict[str, str]]:
    all_docs: List[Dict[str, str]] = []
    current_skiptoken: Optional[int] = get_skiptoken(category)
    
    # If it's the very first run (skiptoken -1), we might want to specifically start from 0
    # or let the API suggest the first skiptoken, which it does via the 'next' link.
    # We'll rely on the 'next' link from the very first response.
    
    initial_api_url = "https://gegevensmagazijn.tweedekamer.nl/SyncFeed/2.0/Feed"
    current_api_url = initial_api_url
    
    while True:
        params = {"category": category}
        if current_skiptoken is not None and current_skiptoken >= 0:
            params["skiptoken"] = current_skiptoken
        
        print(f"--- Fetching API: {current_api_url} with params: {params} ---")
        try:
            resp = requests.get(current_api_url, params=params, timeout=30)
            resp.raise_for_status()
            print(f"API Response Status: {resp.status_code}")
            # print(f"Raw API Response Content (first 500 chars):\n{resp.content.decode('utf-8')[:500]}...") # Keep this commented unless needed
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            break # Exit loop on API error

        root = etree.fromstring(resp.content)
        
        entries = root.findall(".//entry")
        entry_count = len(entries)
        print(f"API returned {entry_count} entries in this batch.")
        
        next_link_found = False
        current_batch_docs = 0 # Counter for docs with enclosures in current batch

        for entry in entries:
            # Extract next skiptoken from link rel="next" if available
            for link in entry.findall("link"): # Check links within entries first for next skiptoken
                if link.get("rel") == "next":
                    href = link.get("href")
                    if "skiptoken=" in href:
                        try:
                            current_skiptoken = int(href.split("skiptoken=")[1].split("&")[0])
                            next_link_found = True
                            print(f"Found next skiptoken in entry: {current_skiptoken}")
                        except ValueError:
                            pass
            
            enclosure_url = None
            for link in entry.findall("link"): # Find enclosure link
                if link.get("rel") == "enclosure":
                    enclosure_url = link.get("href")
                    break
            
            if not enclosure_url:
                # print(f"Entry has no enclosure link. Skipping.") # Uncomment for more verbose skipping
                continue
            
            # Added try-except for enclosure fetching
            try:
                dresp = requests.get(enclosure_url, timeout=30)
                dresp.raise_for_status()
                all_docs.append({"URL": enclosure_url, "content": dresp.text, "Source": "Tweede Kamer"})
                current_batch_docs += 1
            except requests.exceptions.RequestException as e:
                print(f"Error fetching enclosure '{enclosure_url}': {e}")
                continue # Skip this enclosure and try next entry

            if len(all_docs) >= GLOBAL_BATCH_LIMIT:
                print(f"Reached global document limit of {GLOBAL_BATCH_LIMIT}.")
                break # Break if global limit is reached
        
        # Check for next link at the feed level as well, if not found in entries
        if not next_link_found:
            feed_next_link = root.find("{http://www.w3.org/2005/Atom}link[@rel='next']")
            if feed_next_link is not None:
                href = feed_next_link.get("href")
                if "skiptoken=" in href:
                    try:
                        current_skiptoken = int(href.split("skiptoken=")[1].split("&")[0])
                        next_link_found = True
                        print(f"Found next skiptoken in feed level: {current_skiptoken}")
                    except ValueError:
                        pass
        
        if not next_link_found:
            print("No 'next' link found in this feed or any entries. End of feed.")
            break # No more next links, end loop
        
        if len(all_docs) >= GLOBAL_BATCH_LIMIT: # Check limit again after processing entries
            break

    print(f"Collected total of {len(all_docs)} documents.")
    # Save the last successful skiptoken for the next run
    if all_docs: # Only save skiptoken if we actually fetched some docs
        save_skiptoken(category, current_skiptoken) 
    else: # If no docs fetched, but API gives next skiptoken, save it to avoid fetching from -1 again
        if current_skiptoken is not None and current_skiptoken >= 0:
             save_skiptoken(category, current_skiptoken)
             print("Saved the initial next skiptoken to advance for next run.")

    return all_docs


def push_to_hf(docs: List[Dict[str, str]], repo_id: str) -> None:
    if not docs:
        print("No documents to push to Hugging Face.")
        return
    
    print(f"--- Preparing to push {len(docs)} documents to Hugging Face repo: {repo_id} ---")
    api = HfApi()
    
    try:
        # Use Dataset.from_list directly and then its export method
        ds = Dataset.from_list(docs)
        with ds.export("data.csv", format="csv", index=False) as dataset_export:
            api.upload_file(
                path_or_fileobj=dataset_export.path, # Correctly use the path from the export context
                path_in_repo="data/latest.csv",
                repo_id=repo_id,
                repo_type="dataset",
            )
        print(f"Successfully uploaded data to {repo_id}!")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")


if __name__ == "__main__":
    category = "Document"
    
    hf_repo_id = os.getenv("HF_REPO_ID", "user/ingest") # Default if env var not set
    print(f"Hugging Face Repository ID: {hf_repo_id}")
    
    batch = fetch_all_docs(category) # Call the new function
    push_to_hf(batch, hf_repo_id)
