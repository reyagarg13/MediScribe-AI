#!/usr/bin/env python3
"""Basic parsing utilities: regex-based dosage/frequency extraction and fuzzy match placeholder."""
import re
from rapidfuzz import process, fuzz

DOSAGE_PAT = re.compile(r"(\d+\s?(mg|g|mcg|IU|ml))", re.I)
FREQ_PAT = re.compile(r"(once|twice|thrice|\d+\s?times|bd|tds|od|q\d+h|daily|dailyy)", re.I)
DURATION_PAT = re.compile(r"(for\s+\d+\s+(days|day|wk|weeks|week))", re.I)

def extract_dosage(line):
    m = DOSAGE_PAT.search(line)
    return m.group(0) if m else None

def extract_frequency(line):
    m = FREQ_PAT.search(line)
    return m.group(0) if m else None

def extract_duration(line):
    m = DURATION_PAT.search(line)
    return m.group(0) if m else None

def fuzzy_drug_match(token, choices, score_cutoff=70):
    # choices: list[str] of canonical drug names
    match = process.extractOne(token, choices, scorer=fuzz.ratio, score_cutoff=score_cutoff)
    return match  # (name, score, idx) or None
