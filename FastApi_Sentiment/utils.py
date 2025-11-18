import json
import numpy as np
from underthesea import word_tokenize

def normalize_text(s: str):
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s

def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def encode_sentence(sentence, vocab, max_len=150):
    toks = word_tokenize(sentence, format="text").split()
    seq = [vocab.get(t, 1) for t in toks]
    if len(seq) >= max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return toks, np.array([seq], dtype=np.int32)
