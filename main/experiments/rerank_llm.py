# rerank_llm.py

import json, textwrap, pandas as pd
from transformers import pipeline
from kg_driver import driver  # to fetch abstracts

# reuse the same Llama generator from chain_of_thought
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
    # We assume df has columns "pid" and "abstract".
    lines = []
    for _, r in df.iterrows():
        lines.append(f"PID:{r['pid']}")
        # truncate abstract to 600 chars to stay under token limit
        lines.append(f"ABSTRACT:{r['abstract'][:600]}")
        lines.append("")  # blank line between entries
    return "\n".join(lines)

def fetch_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    # If there is no 'pid' column or dataframe is empty, return immediately
    if "pid" not in df.columns or df.empty:
        # Add an 'abstract' column of empty strings so downstream code won't fail
        return df.assign(abstract=[""] * len(df))
    
    ids = list(df["pid"])
    # Run a Cypher query to fetch abstracts for all PIDs
    rows = driver.session().run(
        "MATCH (p:Paper) WHERE p.id IN $ids RETURN p.id AS pid, p.abstract AS abstract",
        ids=ids
    ).data()
    abs_df = pd.DataFrame(rows)  # columns: pid, abstract
    return df.merge(abs_df, on="pid", how="left")

def llm_rerank(paragraph: str, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    1) Fetch abstracts for each candidate (if any).
    2) Build a text block of up to k*4 candidates (in case more).
    3) Send to LLaMA-3 to score each candidate 0-1.
    4) Return top-k by score.
    """
    # 1) Early exit if df is empty
    if df.empty or "pid" not in df.columns:
        df["llm_score"] = []
        return df

    # 2) Ensure we have an 'abstract' column
    df = fetch_abstracts(df)

    # 3) Build the candidate block (only take first k*4 rows to stay under context size)
    block = _mk_block(df.head(k * 4))

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

    # 5) Parse each line that looks like JSON
    scores = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                pid = obj.get("pid")
                score = obj.get("score", 0.0)
                if isinstance(pid, str) and isinstance(score, (int, float)):
                    scores[pid] = float(score)
            except json.JSONDecodeError:
                continue

    # 6) Map scores back to the DataFrame (missing → 0.0)
    df = df.copy()
    df["llm_score"] = df["pid"].map(scores).fillna(0.0)

    # 7) Sort and return top-k
    return df.sort_values("llm_score", ascending=False).head(k)
