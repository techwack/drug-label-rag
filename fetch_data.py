"""Download one FDA label per drug from the openFDA API into data/labels/."""
import json
import time

import requests

from config import DRUGS, LABEL_DIR, SECTIONS

API = "https://api.fda.gov/drug/label.json"


def best_label(results):
    """Pick the label that covers the most of the sections we care about."""
    return max(results, key=lambda r: sum(f in r for f in SECTIONS))


def fetch(drug):
    params = {"search": f'openfda.generic_name:"{drug}"', "limit": 10}
    resp = requests.get(API, params=params, timeout=30)
    resp.raise_for_status()
    label = best_label(resp.json()["results"])
    openfda = label.get("openfda", {})
    return {
        "drug": drug,
        "brand_names": sorted({b.lower() for b in openfda.get("brand_name", [])}),
        "set_id": label.get("set_id"),
        "effective_time": label.get("effective_time"),
        "sections": {f: " ".join(label[f]) for f in SECTIONS if f in label},
    }


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for drug in DRUGS:
        try:
            data = fetch(drug)
        except Exception as e:  # keep going if one drug fails
            print(f"  ! {drug}: {e}")
            continue
        (LABEL_DIR / f"{drug}.json").write_text(json.dumps(data, indent=2))
        print(f"  {drug}: {len(data['sections'])} sections, brands={data['brand_names'][:3]}")
        time.sleep(0.3)  # stay well under openFDA's rate limit


if __name__ == "__main__":
    main()
