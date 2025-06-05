# rerank_llm.py
import json, textwrap, pandas as pd
from transformers import pipeline
from kg_driver import driver          # to fetch abstracts

# reuse the same Llama generator
from chain_of_thought import _gen as _llama

PROMPT = """
You are assessing relevance.  Paragraph:
<<<{para}>>>

For each candidate paper below, output:
{"pid":"<id>","score":<0-1>}

### Candidates
{block}
### End
"""

def _mk_block(df):
    return "\n".join(f"PID:{r.pid}\nABSTRACT:{r.abstract[:600]}"
                     for _,r in df.iterrows())

def fetch_abstracts(df):
    ids=list(df.pid)
    rows=driver.session().run(
        "MATCH (p:Paper)WHERE p.id IN $ids RETURN p.id AS pid,p.abstract AS abstract",
        ids=ids).data()
    return df.merge(pd.DataFrame(rows),on="pid",how="left")

def llm_rerank(paragraph:str, df:pd.DataFrame, k=10):
    df = fetch_abstracts(df)
    block=_mk_block(df.head(k*4))
    raw=_llama(PROMPT.format(para=textwrap.dedent(paragraph),block=block),
               max_new_tokens=400,temperature=0.0,do_sample=False)[0]["generated_text"]
    scores={json.loads(l)["pid"]:json.loads(l)["score"]
            for l in raw.splitlines() if l.strip().startswith("{")}
    df["llm_score"]=df.pid.map(scores).fillna(0)
    return df.sort_values("llm_score",ascending=False).head(k)
