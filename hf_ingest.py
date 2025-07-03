import sqlite3
import requests
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi
import os
from typing import List, Dict, Optional


DB_PATH = "progress.sqlite3"
GLOBAL_BATCH_LIMIT = 1000 # Keep this limit for initial testing, can be increased significantly later


# Define XML Namespaces
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
TK_NAMESPACE = "http://www.tweedekamer.nl/xsd/tkData/v1-0" # Namespace for tk:verwijderd attribute
NAMESPACES = {
    'atom': ATOM_NAMESPACE,
    'tk': TK_NAMESPACE # Added for parsing tk:verwijderd attribute
}


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


def fetch_all_docs(category: str) -> List[Dict[str, str]]:
    all_docs: List[Dict[str, str]] = []
    current_skiptoken: Optional[int] = get_skiptoken(category)
    
    initial_api_url = "https://gegevensmagazijn.tweedekamer.nl/SyncFeed/2.0/Feed"
    current_api_url = initial_api_url
    
    debug_fetch_done = False # Flag to print full raw content only once

    while True:
        params = {"category": category}
        # Only add skiptoken to params if it's not the very first run (which starts at -1)
        # or if we explicitly have a positive skiptoken from a previous 'next' link.
        if current_skiptoken is not None and current_skiptoken >= 0:
            params["skiptoken"] = current_skiptoken
        
        print(f"--- Fetching API: {current_api_url} with params: {params} ---")
        try:
            resp = requests.get(current_api_url, params=params, timeout=30)
            resp.raise_for_status()
            print(f"API Response Status: {resp.status_code}")
            
            # Print full raw content for debugging the first time
            if not debug_fetch_done:
                print(f"Full Raw API Response Content:\n{resp.content.decode('utf-8')}")
                debug_fetch_done = True

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            break # Exit loop on API error

        root = etree.fromstring(resp.content)
        
        # Use namespace prefix for 'entry'
        entries = root.findall("atom:entry", NAMESPACES)
        entry_count = len(entries)
        print(f"API returned {entry_count} entries in this batch.")
        
        next_link_found = False # Reset for current batch
        
        for entry in entries:
            # Extract next skiptoken from link rel="next" if available (can be per entry or at feed level)
            for link in entry.findall("atom:link", NAMESPACES):
                if link.get("rel") == "next":
                    href = link.get("href")
                    if "skiptoken=" in href:
                        try:
                            current_skiptoken = int(href.split("skiptoken=")[1].split("&")[0])
                            next_link_found = True
                            # print(f"Found next skiptoken in entry: {current_skiptoken}") # Too verbose, keep commented
                        except ValueError:
                            pass
            
            # --- NEW LOGIC: Parse nested content XML to check for 'tk:verwijderd' ---
            is_deleted = False
            enclosure_url = None # Initialize enclosure_url here
            
            content_element = entry.find("atom:content", NAMESPACES)
            if content_element is not None and content_element.text is not None:
                try:
                    # Parse the nested XML content string within the <content> tag
                    nested_xml_root = etree.fromstring(content_element.text.encode('utf-8'))
                    
                    # Check for 'tk:verwijderd' attribute on the root element of the nested XML
                    # The tag name (e.g., 'document', 'verslag') might vary, so check any tag at root
                    verwijderd_attr = nested_xml_root.get("{"+TK_NAMESPACE+"}verwijderd")
                    
                    if verwijderd_attr == "true":
                        is_deleted = True
                        entry_id = entry.find("atom:id", NAMESPACES).text if entry.find("atom:id", NAMESPACES) is not None else "N/A"
                        print(f"Skipping entry {entry_id}: marked as deleted (tk:verwijderd='true').")
                except etree.XMLSyntaxError as e:
                    # Log if nested XML is malformed, but continue processing other entries
                    entry_id = entry.find("atom:id", NAMESPACES).text if entry.find("atom:id", NAMESPACES) is not None else "N/A"
                    print(f"Error parsing nested XML content for entry {entry_id}: {e}. Skipping this entry.")
                    continue # Skip this entry if its content XML is invalid
            
            if is_deleted: # If the entry is marked as deleted, skip fetching its enclosure
                continue
            # --- END NEW LOGIC ---

            # Find enclosure link - this part remains the same, but now only runs for non-deleted entries
            for link in entry.findall("atom:link", NAMESPACES):
                if link.get("rel") == "enclosure":
                    enclosure_url = link.get("href")
                    break
            
            if not enclosure_url:
                # print(f"Entry {entry.find('atom:id', NAMESPACES).text if entry.find('atom:id', NAMESPACES) is not None else 'N/A'} has no enclosure link. Skipping.") 
                continue
            
            # Fetch enclosure content
            try:
                dresp = requests.get(enclosure_url, timeout=30)
                dresp.raise_for_status()
                all_docs.append({"URL": enclosure_url, "content": dresp.text, "Source": "Tweede Kamer"})
            except requests.exceptions.RequestException as e:
                print(f"Error fetching enclosure '{enclosure_url}' for entry: {e}. Skipping this enclosure.")
                continue

            # Check global limit after adding each doc
            if len(all_docs) >= GLOBAL_BATCH_LIMIT:
                print(f"Reached global document limit of {GLOBAL_BATCH_LIMIT}.")
                break # Break if global limit is reached

        # Check for next link at the feed level (if not found in entries)
        if not next_link_found:
            feed_next_link = root.find("atom:link[@rel='next']", NAMESPACES)
            if feed_next_link is not None:
                href = feed_next_link.get("href")
                if "skiptoken=" in href:
                    try:
                        current_skiptoken = int(href.split("skiptoken=")[1].split("&")[0])
                        next_link_found = True
                        print(f"Found next skiptoken at feed level: {current_skiptoken}")
                    except ValueError:
                        pass
        
        # If no 'next' link found in the current feed or any entries, it means we've reached the end
        if not next_link_found:
            print("No 'next' link found in this feed or any entries. End of feed.")
            break
        
        if len(all_docs) >= GLOBAL_BATCH_LIMIT:
            break

    print(f"Collected total of {len(all_docs)} documents.")
    # Save the last successful skiptoken. Only save if we advanced, or if we collected some docs.
    # This ensures next run starts from where this one left off or from the next logical point.
    if current_skiptoken is not None:
        save_skiptoken(category, current_skiptoken) 
        print(f"Saved skiptoken for category '{category}': {current_skiptoken}")
    else:
        print("No skiptoken to save for this run.")

    return all_docs


def push_to_hf(docs: List[Dict[str, str]], repo_id: str) -> None:
    if not docs:
        print("No documents to push to Hugging Face.")
        return
    
    print(f"--- Preparing to push {len(docs)} documents to Hugging Face repo: {repo_id} ---")
    api = HfApi()
    
    try:
        local_csv_path = "data.csv" # Define the local path for the CSV
        
        # --- FIX: Use ds.to_csv() to save to a local file ---
        ds = Dataset.from_list(docs)
        ds.to_csv(local_csv_path, index=False) 
        print(f"Successfully saved {len(docs)} documents to {local_csv_path}.")

        api.upload_file(
            path_or_fileobj=local_csv_path, # Use the local file path for upload
            path_in_repo="data/latest.csv", # Destination path in Hugging Face repo
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Successfully uploaded data to {repo_id}!")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")


if __name__ == "__main__":
    category = "Document" 
    
    hf_repo_id = os.getenv("HF_REPO_ID", "user/ingest") 
    print(f"Hugging Face Repository ID: {hf_repo_id}")
    
    batch = fetch_all_docs(category)
    
    push_to_hf(batch, hf_repo_id)
