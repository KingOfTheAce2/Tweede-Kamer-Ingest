import sqlite3
import requests
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi
import os
from typing import List, Dict, Optional
import subprocess # Required for convert_pdf_to_text


DB_PATH = "progress.sqlite3"
# GLOBAL_BATCH_LIMIT = 1000 


# Define XML Namespaces
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
TK_NAMESPACE = "http://www.tweedekamer.nl/xsd/tkData/v1-0" # Namespace for tk:verwijderd attribute
NAMESPACES = {
    'atom': ATOM_NAMESPACE,
    'tk': TK_NAMESPACE
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


def convert_pdf_to_text(pdf_content: bytes) -> str:
    """Converts PDF bytes to plain text using pdftotext."""
    try:
        process = subprocess.run(
            ["pdftotext", "-q", "-", "-"], # -q for quiet, - for stdin, - for stdout
            input=pdf_content, # pdf_content is bytes, which is correct for stdin of external process
            capture_output=True,
            check=True, # Raise an exception for non-zero exit codes
            # Removed: text=True, # <<< REMOVED: This causes the encoding issue with bytes input >>>
            # The encoding and errors are now handled manually on the output
        )
        # <<< MODIFIED: Decode the stdout (which is bytes) into a string >>>
        return process.stdout.decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        # e.stderr will now be bytes, so decode it for printing
        error_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ""
        print(f"Error converting PDF to text: {e.cmd} returned {e.returncode} with output: {error_output}")
        return "" # Return empty string on conversion error
    except Exception as e:
        print(f"Unexpected error during PDF conversion: {e}")
        return ""


def fetch_all_docs(category: str) -> List[Dict[str, str]]:
    all_docs: List[Dict[str, str]] = []
    current_skiptoken: Optional[int] = get_skiptoken(category)
    
    initial_api_url = "https://gegevensmagazijn.tweedekamer.nl/SyncFeed/2.0/Feed"
    current_api_url = initial_api_url
    
    debug_fetch_done = False 

    while True:
        params = {"category": category}
        if current_skiptoken is not None and current_skiptoken >= 0:
            params["skiptoken"] = current_skiptoken
        
        print(f"--- Fetching API: {current_api_url} with params: {params} ---")
        try:
            resp = requests.get(current_api_url, params=params, timeout=30)
            resp.raise_for_status()
            print(f"API Response Status: {resp.status_code}")
            
            if not debug_fetch_done:
                print(f"Full Raw API Response Content:\n{resp.content.decode('utf-8')}") # Print full content for debugging
                debug_fetch_done = True

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            break

        root = etree.fromstring(resp.content)
        
        entries = root.findall("atom:entry", NAMESPACES)
        entry_count = len(entries)
        print(f"API returned {entry_count} entries in this batch.")
        
        next_link_found = False 
        
        for entry in entries:
            entry_id = entry.find("atom:id", NAMESPACES).text if entry.find("atom:id", NAMESPACES) is not None else "N/A"
            
            for link in entry.findall("atom:link", NAMESPACES):
                if link.get("rel") == "next":
                    href = link.get("href")
                    if "skiptoken=" in href:
                        try:
                            current_skiptoken = int(href.split("skiptoken=")[1].split("&")[0])
                            next_link_found = True
                        except ValueError:
                            pass
            
            # --- Filtering for tk:verwijderd is active here (as per your provided code) ---
            is_deleted = False
            enclosure_url = None
            # Do NOT use enclosure_content_type from content_element.get("type") as it's unreliable
            
            content_element = entry.find("atom:content", NAMESPACES)
            if content_element is not None:
                # ONLY use content_element.text for 'is_deleted' check
                if content_element.text is not None:
                    try:
                        nested_xml_root = etree.fromstring(content_element.text.encode('utf-8'))
                        for child_of_nested_root in nested_xml_root:
                            if child_of_nested_root.tag.startswith("{"+TK_NAMESPACE+"}"):
                                verwijderd_attr = child_of_nested_root.get("{"+TK_NAMESPACE+"}verwijderd")
                                if verwijderd_attr == "true":
                                    is_deleted = True
                                    print(f"Skipping entry {entry_id}: marked as deleted (tk:verwijderd='true').")
                                    break
                    except etree.XMLSyntaxError as e:
                        print(f"Error parsing nested XML content for entry {entry_id}: {e}. Skipping this entry.")
                        continue
            
            if is_deleted:
                continue
            # --- End Filtering for tk:verwijderd ---

            for link in entry.findall("atom:link", NAMESPACES):
                if link.get("rel") == "enclosure":
                    enclosure_url = link.get("href")
                    break
            
            if not enclosure_url:
                # print(f"Entry {entry_id} has no enclosure URL. Skipping.")
                continue 
            
            fetched_content = ""
            actual_content_type = "unknown" 
            
            print(f"Processing entry {entry_id} (URL: {enclosure_url})") # Debugging print
            
            try:
                dresp = requests.get(enclosure_url, timeout=30)
                dresp.raise_for_status()
                print(f"Enclosure fetch status for {enclosure_url}: {dresp.status_code}, content_length: {len(dresp.content)} bytes.") # Debugging print

                # <<< MODIFIED LOGIC: Determine content type from HTTP response headers >>>
                if 'Content-Type' in dresp.headers:
                    actual_content_type = dresp.headers['Content-Type'].split(';')[0].strip().lower()
                    print(f"Detected actual content type from headers: {actual_content_type}")
                else:
                    print("Warning: No Content-Type header found in response.")

                if actual_content_type == "application/pdf":
                    print(f"Attempting to convert PDF for {enclosure_url} to text (original size: {len(dresp.content)} bytes)...")
                    fetched_content = convert_pdf_to_text(dresp.content) # Pass raw bytes to converter
                    if not fetched_content.strip(): # Check if text is genuinely empty after stripping whitespace
                        print(f"Warning: PDF {enclosure_url} conversion yielded empty/whitespace text. Skipping.")
                        continue # Skip if PDF conversion fails
                    print(f"PDF converted successfully. Text length: {len(fetched_content)} characters.")
                elif actual_content_type.startswith("text/") or actual_content_type == "application/xml":
                    fetched_content = dresp.text
                    print(f"Fetched text/xml content (length: {len(fetched_content)} characters).")
                elif actual_content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    # Implement conversion for Word documents here if needed.
                    # For now, they will be skipped, as per your previous logs.
                    print(f"Skipping unsupported content type: {actual_content_type} for {enclosure_url} (raw content size: {len(dresp.content)} bytes).")
                    continue
                else:
                    print(f"Skipping unrecognized content type: {actual_content_type} for {enclosure_url} (raw content size: {len(dresp.content)} bytes).")
                    continue
                
                all_docs.append({"URL": enclosure_url, "content": fetched_content, "Source": "Tweede Kamer"})

            except requests.exceptions.RequestException as e:
                print(f"Error fetching enclosure '{enclosure_url}': {e}. Skipping this enclosure.")
                continue

        #     if len(all_docs) >= GLOBAL_BATCH_LIMIT:
        #         print(f"Reached global document limit of {GLOBAL_BATCH_LIMIT}.")
        #         break

        # if len(all_docs) >= GLOBAL_BATCH_LIMIT: 
        #     break
        
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
        
        if not next_link_found:
            print("No 'next' link found in this feed or any entries. End of feed.")
            break
        
    print(f"Collected total of {len(all_docs)} documents.")
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
        local_parquet_path = "data.parquet" # Changed to Parquet
        
        ds = Dataset.from_list(docs)
        ds.to_parquet(local_parquet_path) 
        print(f"Successfully saved {len(docs)} documents to {local_parquet_path}.")

        api.upload_file(
            path_or_fileobj=local_parquet_path,
            path_in_repo="data/latest.parquet", # Changed destination path
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
