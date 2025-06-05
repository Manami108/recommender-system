# rerank_llm.py

import json
import textwrap
import pandas as pd
from transformers import pipeline
from kg_driver import driver  # to fetch abstracts

# Reuse the same Llama generator from chain_of_thought:
from chain_of_thought import _gen as _llama

PROMPT = """
You are assessing relevance.  Paragraph:
<<<{para}>>>

For each candidate paper below, output exactly one line of JSON:
{"pid":"<id>","score":<0-1>}

### Candidates
{block}
### End
"""

def _mk_block(df: pd.DataFrame) -> str:
    """
    Build a text block for up to k*4 candidates.  
    Each candidate turns into two lines:
      PID:<pid>
      ABSTRACT:<first 600 chars of abstract>
    """
    lines = []
    for _, r in df.iterrows():
        # r["pid"] should exist
        lines.append(f"PID:{r['pid']}")
        # r["abstract"] is guaranteed a string (we fill NaN → "")
        abstract = r["abstract"]
        # truncate to 600 chars
        snippet = abstract[:600] if isinstance(abstract, str) else ""
        lines.append(f"ABSTRACT:{snippet}")
        lines.append("")  # blank line between entries
    return "\n".join(lines)

def fetch_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing at least a 'pid' column, fetch each abstract from Neo4j.
    If df is empty or has no 'pid', return df with an 'abstract' column of empty strings.
    Otherwise, merge on pid and fill any missing abstracts with "".
    """
    # Early exit: if no 'pid' column or empty DataFrame, just add empty abstracts
    if "pid" not in df.columns or df.empty:
        return df.assign(abstract=[""] * len(df))

    ids = list(df["pid"])
    # Pull all abstracts from Neo4j for these PIDs
    result_rows = driver.session().run(
        "MATCH (p:Paper) WHERE p.id IN $ids RETURN p.id AS pid, p.abstract AS abstract",
        ids=ids
    ).data()

    abstracts_df = pd.DataFrame(result_rows)  # columns: pid, abstract
    # Merge so that any pid not returned from Neo4j becomes NaN → we fill with ""
    merged = df.merge(abstracts_df, on="pid", how="left")
    merged["abstract"] = merged["abstract"].fillna("")  # fill missing abstracts
    return merged

def llm_rerank(paragraph: str, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    1) If df is empty or has no 'pid', return immediately (with an 'llm_score' column).
    2) Fetch abstracts, fill any missing values with "".
    3) Build a block of up to k*4 candidates (truncate abstract to 600 chars each).
    4) Send to Llama-3 to get a score for each PID.
    5) Map scores back and return top-k by LLM score.
    """
    # 1) Early exit if no candidates
    if df.empty or "pid" not in df.columns:
        df = df.copy()
        df["llm_score"] = 0.0
        return df

    # 2) Ensure we have a string 'abstract' column for each row
    df_with_abs = fetch_abstracts(df)

    # 3) Build the candidate block using only the first k*4 rows
    block = _mk_block(df_with_abs.head(k * 4))

    # 4) Run LLaMA scoring prompt
    prompt = PROMPT.format(
        para=textwrap.dedent(paragraph).strip(),
        block=block
    )
    raw = _llama(
        prompt,
        max_new_tokens=400,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    # 5) Parse each JSON‐looking line into { "pid": <id>, "score": <float> }
    scores = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                pid = obj.get("pid")
                score = obj.get("score", 0.0)
                if isinstance(pid, str) and isinstance(score, (int, float)):
                    scores[pid] = float(score)
            except json.JSONDecodeError:
                # If one line isn’t valid JSON, just skip it
                continue

    # 6) Map the scores back to the DataFrame, defaulting missing to 0.0
    df_scored = df_with_abs.copy()
    df_scored["llm_score"] = df_scored["pid"].map(scores).fillna(0.0)

    # 7) Sort by llm_score descending and take top-k
    return df_scored.sort_values("llm_score", ascending=False).head(k)
