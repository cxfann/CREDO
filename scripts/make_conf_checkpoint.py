#!/usr/bin/env python
"""Build models/Qwen3-8B-conf = Qwen3-8B + <CONF_HIGH>/<CONF_LOW> reserved tokens.

The two reserved tokens are added into the tokenizer's existing embedding headroom, so
no resize is needed. Qwen3-8B has tie_word_embeddings=False, so both the input-embedding
and the lm_head rows are initialized. Each reserved token is *copy-initialized* from a
single-token anchor word (copyinit): <CONF_HIGH> copies the row of "high" and <CONF_LOW>
copies the row of "low", on both the embedding and the lm_head side. Copying (rather than
averaging several anchors) gives the readout a usable, well-separated step-0 signal; see
Appendix B.

Run (CPU):
  CONF_CKPT_SRC=models/Qwen3-8B CONF_CKPT_DST=models/Qwen3-8B-conf \
      python scripts/make_conf_checkpoint.py
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.environ.get("CONF_CKPT_SRC", "models/Qwen3-8B")
DST = os.environ.get("CONF_CKPT_DST", "models/Qwen3-8B-conf")
HIGH, LOW = "<CONF_HIGH>", "<CONF_LOW>"
# Each reserved token copy-inits from a single-token anchor word (no leading space).
ANCHOR = {HIGH: "high", LOW: "low"}


def _single_token_id(tok, word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"anchor {word!r} is not a single token: {ids}"
    return ids[0]


def main():
    tok = AutoTokenizer.from_pretrained(SRC)
    n0 = len(tok)
    added = tok.add_special_tokens({"additional_special_tokens": [HIGH, LOW]})
    assert added == 2, f"expected to add 2 tokens, got {added}"
    id_high, id_low = tok.convert_tokens_to_ids(HIGH), tok.convert_tokens_to_ids(LOW)
    print(f"tokenizer: {n0} -> {len(tok)}; id_high={id_high} id_low={id_low}")
    assert (id_high, id_low) == (151669, 151670), "unexpected token ids (is SRC the stock Qwen3-8B?)"

    model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16)
    assert model.config.vocab_size >= len(tok), "resize needed (unexpected: no embedding headroom)"
    assert model.config.tie_word_embeddings is False, "expected untied embeddings for Qwen3-8B"

    emb = model.get_input_embeddings().weight   # (vocab, hidden)
    lmh = model.get_output_embeddings().weight  # (vocab, hidden)
    report = {"src": SRC, "id_high": id_high, "id_low": id_low, "resize": False, "anchors": {}}
    with torch.no_grad():
        for conf_tok, anchor_word in ANCHOR.items():
            cid = tok.convert_tokens_to_ids(conf_tok)
            aid = _single_token_id(tok, anchor_word)
            # copyinit: clone the anchor row into the reserved-token row, on both sides
            emb[cid] = emb[aid].clone()
            lmh[cid] = lmh[aid].clone()
            report["anchors"][conf_tok] = {"word": anchor_word, "anchor_id": aid}
            assert torch.equal(emb[cid], emb[aid]) and torch.equal(lmh[cid], lmh[aid])

    model.save_pretrained(DST)
    tok.save_pretrained(DST)

    # reload assertions: the two reserved rows equal their anchors byte-for-byte
    tok2 = AutoTokenizer.from_pretrained(DST)
    m2 = AutoModelForCausalLM.from_pretrained(DST, torch_dtype=torch.bfloat16)
    e2, l2 = m2.get_input_embeddings().weight, m2.get_output_embeddings().weight
    for conf_tok, anchor_word in ANCHOR.items():
        cid, aid = tok2.convert_tokens_to_ids(conf_tok), _single_token_id(tok2, anchor_word)
        assert torch.equal(e2[cid], e2[aid]) and torch.equal(l2[cid], l2[aid]), conf_tok
    enc = tok2("<confidence><CONF_HIGH></confidence>", add_special_tokens=False)["input_ids"]
    assert tok2.convert_tokens_to_ids(HIGH) in enc, f"literal encode failed: {enc}"
    assert HIGH in tok2.decode(enc, skip_special_tokens=False)
    assert HIGH not in tok2.decode(enc, skip_special_tokens=True)

    print(json.dumps(report, indent=2))
    print("CONF_CHECKPOINT_ASSERTIONS: PASS")


if __name__ == "__main__":
    sys.exit(main())
