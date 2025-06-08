
# --------------------------- config.py -----------------------
"""Shared configuration, lazy loaders, Neo4j + models."""
import os, re, json
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

LLAMA_MODEL          = "meta-llama/Meta-Llama-3-8B-Instruct"
SLIDING_WIN_TOKENS   = 128
SLIDING_STRIDE       = 64
EMBED_TOP_K          = 40
KEYWORD_TOP_K        = 40
MAX_TOTAL_CANDS      = 120
CITE_HOPS            = 2
CITE_EXPAND_K        = 60
FINAL_LLM_K          = 15
PROMPT_DIR           = Path(__file__).parent / "prompts"

# ---- lazy singletons ---------------------------------------
@lru_cache()
def tokenizer():
    return AutoTokenizer.from_pretrained(LLAMA_MODEL)

@lru_cache()
def llama_gen():
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype="auto", device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer())

@lru_cache()
def embedder():
    em = SentenceTransformer("allenai/scibert_scivocab_uncased")
    em.eval(); return em

@lru_cache()
def driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"),
              os.getenv("NEO4J_PASS", "Manami1008")))