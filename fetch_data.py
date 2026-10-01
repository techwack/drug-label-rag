"""Download one FDA label per drug from the openFDA API into data/labels/."""
import json
import time

import requests

from config import BRANDS, DRUGS, LABEL_DIR, SECTIONS

API = "https://api.fda.gov/drug/label.json"


def is_single_ingredient(label):
    """True for labels with one active ingredient (skips combos like amlodipine + benazepril)."""
    return len(label.get("openfda", {}).get("substance_name", [])) == 1


def best_label(results):
    """Prefer single-ingredient labels, then the one covering the most sections we care about."""
    single = [r for r in results if is_single_ingredient(r)]
    return max(single or results, key=lambda r: sum(f in r for f in SECTIONS))


def fetch(drug):
    params = {"search": f'openfda.generic_name:"{drug}"', "limit": 50}
    resp = requests.get(API, params=params, timeout=30)
    resp.raise_for_status()
    label = best_label(resp.json()["results"])
    openfda = label.get("openfda", {})
    return {
        "drug": drug,
        # Generic manufacturers' labels list the generic name as the "brand",
        # so add the well-known brand names from config as well.
        "brand_names": sorted({b.lower() for b in openfda.get("brand_name", [])} | set(BRANDS.get(drug, []))),
        "single_ingredient": is_single_ingredient(label),
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
        single = "" if data["single_ingredient"] else "  (combination product!)"
        print(f"  {drug}: {len(data['sections'])} sections, brands={data['brand_names'][:4]}{single}")
        time.sleep(0.3)  # stay well under openFDA's rate limit


if __name__ == "__main__":
    main()
