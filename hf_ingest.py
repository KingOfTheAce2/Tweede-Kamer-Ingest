import sqlite3
import requests
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi
from typing import List, Dict

DB_PATH = "progress.sqlite3"
BATCH_SIZE = 100


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


def fetch_batch(category: str) -> List[Dict[str, str]]:
    token = get_skiptoken(category)
    url = "https://gegevensmagazijn.tweedekamer.nl/SyncFeed/2.0/Feed"
    params = {"category": category}
    if token >= 0:
        params["skiptoken"] = token
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    root = etree.fromstring(resp.content)
    docs: List[Dict[str, str]] = []
    next_token = token
    for entry in root.findall(".//entry"):
        # track next skiptoken
        for link in entry.findall("link"):
            if link.get("rel") == "next":
                href = link.get("href")
                if "skiptoken=" in href:
                    try:
                        next_token = int(href.split("skiptoken=")[1].split("&")[0])
                    except ValueError:
                        pass
        enclosure = None
        for link in entry.findall("link"):
            if link.get("rel") == "enclosure":
                enclosure = link.get("href")
                break
        if not enclosure:
            continue
        dresp = requests.get(enclosure, timeout=30)
        if dresp.status_code == 200:
            docs.append({"URL": enclosure, "content": dresp.text, "Source": "Tweede Kamer"})
        if len(docs) >= BATCH_SIZE:
            break
    save_skiptoken(category, next_token)
    return docs


def push_to_hf(docs: List[Dict[str, str]], repo_id: str) -> None:
    if not docs:
        return
    ds = Dataset.from_list(docs)
    api = HfApi()
    with ds.export(
        "data.csv", format="csv", index=False
    ):  # export dataset to a CSV file
        api.upload_file(
            path_or_fileobj="data.csv",
            path_in_repo="data/latest.csv",
            repo_id=repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    category = "Document"
    batch = fetch_batch(category)
    push_to_hf(batch, "your-username/tweede-kamer-batch")
