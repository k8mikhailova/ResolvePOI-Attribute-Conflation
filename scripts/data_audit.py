# data_audit.py
import json
import re
from urllib.parse import urlparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# -----------------------
# Config
# -----------------------
PARQUET_PATH = "project_A_data.parquet"   # change if your file name differs
SAMPLE_N = 2000                           # set to None to analyze full dataset (can be heavy)


# -----------------------
# Helpers: parsing + normalization
# -----------------------
def parse_maybe_json(x):
    """Parse JSON strings into Python objects; return original if not parseable."""
    if pd.isna(x):
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Avoid json.loads on ordinary strings
        if (s[:1] in "{[" and s[-1:] in "}]") or s.lower() in ("null", "true", "false"):
            try:
                return json.loads(s)
            except Exception:
                return x
        return x
    return x


def is_effectively_empty(x):
    """Treat None/NaN/empty string/empty list/empty dict as missing."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    if isinstance(x, (list, dict)) and len(x) == 0:
        return True
    return False


def normalize_url(u: str) -> str:
    """Normalize URL-ish strings for comparison: host+path (no scheme/query)."""
    if u is None:
        return ""
    if not isinstance(u, str):
        u = str(u)
    u = u.strip()
    if not u:
        return ""
    # Add scheme if missing for parsing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u
    try:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        path = (p.path or "").rstrip("/")
        if host.startswith("www."):
            host = host[4:]
        out = (host + path).lower()
        return out if host else ""
    except Exception:
        return ""


def normalize_phone(p: str) -> str:
    """Keep digits, preserve leading + if present. Not full E.164 formatting."""
    if p is None:
        return ""
    if not isinstance(p, str):
        p = str(p)
    p = p.strip()
    if not p:
        return ""
    plus = p.startswith("+")
    digits = re.sub(r"\D", "", p)
    if not digits:
        return ""
    return ("+" if plus else "") + digits


def extract_emails(value):
    v = parse_maybe_json(value)
    found = set()
    email_re = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

    def scan(s):
        if not s:
            return
        for m in email_re.findall(str(s)):
            found.add(m.lower())

    if isinstance(v, str):
        scan(v)
    elif isinstance(v, list):
        for item in v:
            scan(item)
    elif isinstance(v, dict):
        for item in v.values():
            scan(item)
    return found


def extract_socials(value):
    """Extract socials from strings/lists/dicts into normalized handles/urls."""
    v = parse_maybe_json(value)
    out = set()

    def add_one(s):
        if not s:
            return
        s = str(s).strip()
        if not s:
            return
        if s.startswith("@"):
            out.add(s.lower())
            return
        norm = normalize_url(s)
        if norm:
            out.add(norm)
            return
        out.add(s.lower())

    if isinstance(v, str):
        add_one(v)
    elif isinstance(v, list):
        for item in v:
            add_one(item)
    elif isinstance(v, dict):
        # often platform -> url/handle
        for k, item in v.items():
            add_one(item)
            # sometimes keys contain useful info too
            add_one(k)
    return out


def extract_primary(value):
    """Extract 'primary' if JSON dict, else string."""
    v = parse_maybe_json(value)
    if isinstance(v, dict):
        primary = v.get("primary")
        return primary.strip() if isinstance(primary, str) else ""
    return value.strip() if isinstance(value, str) else ""


def value_signature(attr: str, value):
    """
    Convert an attribute value into a comparable signature to detect conflicts robustly.
    """
    if is_effectively_empty(value):
        return ""

    if attr in ("names", "categories", "addresses"):
        return extract_primary(value).lower()

    if attr == "websites":
        v = parse_maybe_json(value)
        vals = []
        if isinstance(v, str):
            vals = [v]
        elif isinstance(v, list):
            vals = v
        elif isinstance(v, dict):
            if "primary" in v:
                vals.append(v["primary"])
            if isinstance(v.get("alternate"), list):
                vals.extend(v["alternate"])
            else:
                vals.extend(list(v.values()))
        normed = sorted({normalize_url(str(x)) for x in vals if normalize_url(str(x))})
        return tuple(normed)

    if attr == "phones":
        v = parse_maybe_json(value)
        vals = []
        if isinstance(v, str):
            vals = [v]
        elif isinstance(v, list):
            vals = v
        elif isinstance(v, dict):
            if "primary" in v:
                vals.append(v["primary"])
            if isinstance(v.get("alternate"), list):
                vals.extend(v["alternate"])
            else:
                vals.extend(list(v.values()))
        normed = sorted({normalize_phone(str(x)) for x in vals if normalize_phone(str(x))})
        return tuple(normed)

    if attr == "socials":
        return tuple(sorted(extract_socials(value)))

    if attr == "emails":
        return tuple(sorted(extract_emails(value)))

    return str(value).strip().lower()


# -----------------------
# Audit functions
# -----------------------
def print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def duckdb_load_parquet(path: str) -> pd.DataFrame:
    """
    Load parquet via DuckDB (no pyarrow needed) and return as pandas DataFrame.
    Note: .df() conversion uses pandas/numpy only.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p.resolve()}")

    # Using DuckDB to read parquet
    return duckdb.query(f"SELECT * FROM '{p.as_posix()}'").df()


def audit(df: pd.DataFrame):
    print_header("BASIC SHAPE")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Columns:\n", list(df.columns))

    # Find current/base pairs
    attr_pairs = []
    for c in df.columns:
        if c.startswith("base_"):
            curr = c.replace("base_", "", 1)
            if curr in df.columns:
                attr_pairs.append((curr, c))

    print_header("CURRENT/BASE PAIRS FOUND")
    if attr_pairs:
        for curr, base in attr_pairs:
            print(f"- {curr} <-> {base}")
    else:
        print("No base_ pairs detected.")

    # Missingness
    print_header("MISSINGNESS (treat empty strings/lists/dicts as missing)")
    missing_counts = {c: int(df[c].map(is_effectively_empty).sum()) for c in df.columns}
    miss = (
        pd.DataFrame(
            {
                "missing": pd.Series(missing_counts),
                "missing_pct": pd.Series({k: v / len(df) for k, v in missing_counts.items()}),
            }
        )
        .sort_values("missing_pct", ascending=False)
    )
    print(miss.head(25).to_string())
    miss.to_csv("audit_missingness.csv", index=True)
    print("\nSaved: audit_missingness.csv")

    # Type inspection
    print_header("FORMAT / TYPE INSPECTION (types seen in first ~200 non-null rows)")
    for c in df.columns[:40]:  # limit printing to not explode your terminal
        sample = df[c].dropna().head(200)
        types = sample.map(lambda x: type(x).__name__).value_counts().to_dict()
        json_like = 0
        if sample.dtype == "object":
            for x in sample:
                if isinstance(x, str):
                    s = x.strip()
                    if s.startswith("{") or s.startswith("["):
                        json_like += 1
        print(f"{c}: types={types} | json_like_strings_in_sample={json_like}")

    # Confidence sanity
    if "confidence" in df.columns:
        print_header("CONFIDENCE DISTRIBUTION (current)")
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        print(conf.describe().to_string())
        weird = df[conf.isna() | (conf < 0) | (conf > 1)]
        print("Weird confidence rows (NaN or outside [0,1]):", len(weird))
        if len(weird) > 0:
            weird.head(50).to_csv("audit_confidence_weird.csv", index=False)
            print("Saved sample: audit_confidence_weird.csv")

    if "base_confidence" in df.columns:
        print_header("CONFIDENCE DISTRIBUTION (base)")
        bconf = pd.to_numeric(df["base_confidence"], errors="coerce")
        print(bconf.describe().to_string())

    # Conflict rates
    print_header("CURRENT vs BASE CONFLICT RATES (normalized comparison)")
    key_attrs = ["names", "categories", "websites", "phones", "socials", "emails", "addresses"]
    rows = []

    for attr in key_attrs:
        base_attr = f"base_{attr}"
        if attr in df.columns and base_attr in df.columns:
            sig_curr = df[attr].map(lambda v: value_signature(attr, v))
            sig_base = df[base_attr].map(lambda v: value_signature(attr, v))

            both_missing = df[attr].map(is_effectively_empty) & df[base_attr].map(is_effectively_empty)
            one_missing = df[attr].map(is_effectively_empty) ^ df[base_attr].map(is_effectively_empty)
            same = (~both_missing) & (sig_curr == sig_base)
            conflict = (~both_missing) & (~same)

            rows.append(
                {
                    "attribute": attr,
                    "both_missing_pct": both_missing.mean(),
                    "one_missing_pct": one_missing.mean(),
                    "same_pct": same.mean(),
                    "conflict_pct": conflict.mean(),
                }
            )

            # Save conflict examples
            ex = df[conflict].copy().head(50)
            if len(ex) > 0 and "id" in df.columns:
                ex[["id", attr, base_attr]].to_csv(f"audit_conflicts_{attr}.csv", index=False)

    if rows:
        conflict_df = pd.DataFrame(rows).sort_values("conflict_pct", ascending=False)
        print(conflict_df.to_string(index=False))
        conflict_df.to_csv("audit_conflict_rates.csv", index=False)
        print("\nSaved: audit_conflict_rates.csv and audit_conflicts_<attr>.csv samples")
    else:
        print("No current/base pairs found among:", key_attrs)

    # Quick quality checks (sample-based)
    print_header("QUALITY CHECKS (sample-based): websites / phones / emails")
    checks = []

    def sample_tokens(series, n=500):
        return series.dropna().head(n)

    if "websites" in df.columns:
        total = 0
        bad = 0
        for x in sample_tokens(df["websites"]):
            v = parse_maybe_json(x)
            vals = []
            if isinstance(v, str):
                vals = [v]
            elif isinstance(v, list):
                vals = v
            elif isinstance(v, dict):
                vals = list(v.values())
            for u in vals:
                total += 1
                if not normalize_url(str(u)):
                    bad += 1
        checks.append(("websites url tokens (sample)", total, bad, (bad / total) if total else np.nan))

    if "phones" in df.columns:
        total = 0
        bad = 0
        for x in sample_tokens(df["phones"]):
            v = parse_maybe_json(x)
            vals = []
            if isinstance(v, str):
                vals = [v]
            elif isinstance(v, list):
                vals = v
            elif isinstance(v, dict):
                vals = list(v.values())
            for p in vals:
                total += 1
                nrm = normalize_phone(str(p))
                digits = re.sub(r"\D", "", nrm)
                if (not digits) or (len(digits) < 7):
                    bad += 1
        checks.append(("phones tokens too short/empty (sample)", total, bad, (bad / total) if total else np.nan))

    if "emails" in df.columns:
        total = 0
        bad = 0
        for x in sample_tokens(df["emails"]):
            total += 1
            if len(extract_emails(x)) == 0:
                bad += 1
        checks.append(("email rows with no detectable email (sample)", total, bad, (bad / total) if total else np.nan))

    if checks:
        qc = pd.DataFrame(checks, columns=["check", "sample_size", "bad_count", "bad_rate"])
        print(qc.to_string(index=False))
        qc.to_csv("audit_quality_checks.csv", index=False)
        print("\nSaved: audit_quality_checks.csv")
    else:
        print("No websites/phones/emails columns found for quality checks.")


def main():
    print_header("LOADING PARQUET VIA DUCKDB")
    df = duckdb_load_parquet(PARQUET_PATH)

    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
        print(f"Using SAMPLE_N={SAMPLE_N} rows for audit (random_state=42).")
    else:
        print("Using full dataset for audit.")

    audit(df)


if __name__ == "__main__":
    main()