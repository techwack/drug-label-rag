"""Shared settings for the drug-label RAG pipeline."""
from pathlib import Path

ROOT = Path(__file__).parent
LABEL_DIR = ROOT / "data" / "labels"
INDEX_DIR = ROOT / "index"
EVAL_DIR = ROOT / "eval"

LLM_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 150
TOP_K = 5

DRUGS = [
    "warfarin", "metformin", "lisinopril", "atorvastatin", "amoxicillin",
    "ibuprofen", "acetaminophen", "sertraline", "omeprazole", "amlodipine",
    "levothyroxine", "prednisone", "gabapentin", "simvastatin", "losartan",
    "clopidogrel", "albuterol", "fluoxetine", "hydrochlorothiazide", "methotrexate",
]

# Well-known brand names, so questions like "What does the Plavix label say?" find clopidogrel.
BRANDS = {
    "warfarin": ["coumadin", "jantoven"], "metformin": ["glucophage"],
    "lisinopril": ["zestril", "prinivil"], "atorvastatin": ["lipitor"],
    "amoxicillin": ["amoxil"], "ibuprofen": ["advil", "motrin"],
    "acetaminophen": ["tylenol"], "sertraline": ["zoloft"], "omeprazole": ["prilosec"],
    "amlodipine": ["norvasc"], "levothyroxine": ["synthroid", "levoxyl"],
    "gabapentin": ["neurontin"], "simvastatin": ["zocor"], "losartan": ["cozaar"],
    "clopidogrel": ["plavix"], "albuterol": ["ventolin", "proair"],
    "fluoxetine": ["prozac"], "hydrochlorothiazide": ["microzide"],
    "methotrexate": ["trexall", "otrexup"],
}

# openFDA field -> readable section title
SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "overdosage": "Overdosage",
    "mechanism_of_action": "Mechanism of Action",
}
