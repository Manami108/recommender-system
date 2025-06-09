from pathlib import Path
import json
import pandas as pd
from context_analysis import analyze_paragraph_context

SCORE_TMPL = Path("prompts/coherence_score_with_context.prompt").read_text()

def llm_contextual_rerank(paragraph: str, candidates: pd.DataFrame, k: int=10) -> pd.DataFrame:
    # 1) Analyze the paragraph once
    ctx = analyze_paragraph_context(paragraph)
    ctx_json = json.dumps(ctx, ensure_ascii=False)

    # 2) Build the candidates block
    cand_lines = []
    for _, r in candidates.iterrows():
        cand_lines.append(f"{r.pid}\nTitle: {r.title}\nAbstract: {r.abstract.strip()}")
    cand_block = "\n\n".join(cand_lines)

    # 3) Fill the scoring prompt
    prompt = (SCORE_TMPL
              .replace("<<<CONTEXT_JSON>>>", ctx_json)
              .replace("<<<CANDIDATES>>>", cand_block))

    # 4) Call LLaMA
    raw = gen(prompt)[0]["generated_text"].split("<END>")[0]
    # Extract the JSON array
    start, end = raw.find("["), raw.rfind("]")+1
    arr = json.loads(raw[start:end])

    # 5) Merge back & sort
    score_df = pd.DataFrame(arr)
    ranked = candidates.merge(score_df, on="pid")
    return ranked.sort_values("final_score", ascending=False).head(k)
