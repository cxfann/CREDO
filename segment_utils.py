#!/usr/bin/env python
"""Segment utilities: char->token mapping and segment/readout detection.

Design:
- boundary = index of the first completion token that contains the first char of
  the FIRST "<analysis>" occurrence in the decoded text. Tokens [0, boundary) are
  the answer segment; [boundary, T) the confidence segment.
- readout_pos = index of the token right after the "<confidence>" opener
  (fast path: the special token id itself if present exactly once).
- Mapping uses binary search over prefix-decode lengths: O(log T) decodes per
  lookup (NOT per-token incremental decode). decode is done with
  skip_special_tokens=False so special tokens keep their literal chars.
"""
from typing import List, Optional, Tuple

ANALYSIS_TAG = "<analysis>"
CONF_OPEN = "<confidence>"


def _decode(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def token_index_for_char(tokenizer, ids: List[int], char_pos: int, full_text: Optional[str] = None) -> Optional[int]:
    """Smallest j such that len(decode(ids[:j+1])) > char_pos, i.e. token containing char_pos."""
    if full_text is None:
        full_text = _decode(tokenizer, ids)
    if char_pos >= len(full_text):
        return None
    lo, hi = 0, len(ids) - 1  # invariant: answer in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_decode(tokenizer, ids[: mid + 1])) > char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def find_segments(tokenizer, ids: List[int], high_id: Optional[int] = None, low_id: Optional[int] = None) -> Tuple[str, Optional[int], Optional[int]]:
    """Return (decoded_text_with_specials, boundary_idx_or_None, readout_idx_or_None).

    readout detection: fast path = unique special token position; fallback =
    char position right after the first "<confidence>" opener that appears at or
    after the boundary. If no opener, readout is None.
    """
    text = _decode(tokenizer, ids)
    b_char = text.find(ANALYSIS_TAG)
    boundary = token_index_for_char(tokenizer, ids, b_char, text) if b_char >= 0 else None

    readout = None
    if high_id is not None and low_id is not None:
        hits = [i for i, t in enumerate(ids) if t == high_id or t == low_id]
        if len(hits) == 1:
            readout = hits[0]
    if readout is None:
        search_from = b_char if b_char >= 0 else 0
        c_char = text.find(CONF_OPEN, search_from)
        if c_char >= 0:
            after = c_char + len(CONF_OPEN)
            readout = token_index_for_char(tokenizer, ids, after, text)
    return text, boundary, readout


def compute_channel_advantages(acc_g, conf_g, F_g, G, method,
                               cal_gamma=0.5, cal_weight=0.5, w_fmt_ans=0.5, mse_gamma=None,
                               ans_reward_mode="fmt_gated_acc",
                               sw_kappa=0.0, sw_kappa_neg=None, sw_valid_g=None,
                               sw_weight_clip=(0.5, 2.0)):
    """Pure channel math for dcpo/credo segmented scoring.

    All inputs are 1-D float tensors of length num_groups*G, grouped contiguously.
    Returns dict with target_g, r_ans_g, r_conf_g, adv_ans_g, adv_conf_g, mse_target_g, sw_w_g.
    Group normalization: (x - group_mean) / (group_std + 1e-4), Bessel std (torch default).

    ans_reward_mode (CREDO only):
      fmt_gated_acc = w_fmt_ans*F + F*acc   (function default)
      fmt_plus_acc  = w_fmt_ans*F + acc     (ungated acc, additive format term)
      acc_only      = acc                   (no answer-channel format pressure)
    By affine invariance of the group norm the three coincide on all-F=1 groups;
    they differ only through F=0 samples (truncated / malformed).

    SW / surprise weighting (CREDO only, dormant by default):
      s_i = |acc_i - c_i| on readout-valid samples (sw_valid_g, e.g. vLLM top-k pair coverage);
      w_i = clip(1 + kappa*(s_i - group_mean_valid(s)), w_min, w_max); invalid rows get w=1.
      adv_ans_g *= w_i  — the ONLY effect; rewards, conf channel and mse_target untouched.
      Weight is applied AFTER group norm, so group statistics never see w. sw_kappa=0.0
      skips the whole block (bit-identical output). kappa_neg (if not None) is used where
      adv_ans<0 (asymmetric variant). w_min>0 guarantees the advantage sign never flips.
    """
    import torch

    group_acc = acc_g.view(-1, G).mean(dim=1).repeat_interleave(G, dim=0)
    target_g = cal_gamma * acc_g + (1.0 - cal_gamma) * group_acc
    if method == "dcpo":
        r_ans_g = acc_g.clone()
        r_conf_g = -cal_weight * (target_g - conf_g) ** 2
    elif method == "credo":
        if ans_reward_mode == "fmt_gated_acc":
            r_ans_g = w_fmt_ans * F_g + F_g * acc_g
        elif ans_reward_mode == "fmt_plus_acc":
            r_ans_g = w_fmt_ans * F_g + acc_g
        elif ans_reward_mode == "acc_only":
            r_ans_g = acc_g.clone()
        else:
            raise ValueError(f"invalid ans_reward_mode: {ans_reward_mode}")
        r_conf_g = F_g * (1.0 - (target_g - conf_g) ** 2)
    else:
        raise ValueError(f"segmented scoring does not support method={method}")

    def _gnorm(x):
        m = x.view(-1, G).mean(dim=1).repeat_interleave(G, dim=0)
        s = x.view(-1, G).std(dim=1).repeat_interleave(G, dim=0)
        return (x - m) / (s + 1e-4)

    adv_ans_g = _gnorm(r_ans_g)
    adv_conf_g = _gnorm(r_conf_g)

    sw_w_g = None
    if float(sw_kappa) > 0.0:
        if method != "credo":
            raise ValueError(f"sw_kappa > 0 requires method=credo (got {method})")
        v = sw_valid_g if sw_valid_g is not None else torch.ones_like(acc_g)
        v = v.to(acc_g.dtype)
        s = (acc_g - conf_g).abs() * v
        n_valid = v.view(-1, G).sum(dim=1).clamp(min=1.0)
        s_bar = (s.view(-1, G).sum(dim=1) / n_valid).repeat_interleave(G, dim=0)
        kap_pos = float(sw_kappa)
        kap_neg = float(sw_kappa_neg) if sw_kappa_neg is not None else kap_pos
        kap = torch.where(adv_ans_g >= 0,
                          torch.full_like(adv_ans_g, kap_pos),
                          torch.full_like(adv_ans_g, kap_neg))
        w = (1.0 + kap * (s - s_bar)).clamp(float(sw_weight_clip[0]), float(sw_weight_clip[1]))
        w = torch.where(v > 0, w, torch.ones_like(w))
        adv_ans_g = adv_ans_g * w
        sw_w_g = w

    mg = mse_gamma if mse_gamma is not None else cal_gamma
    return {
        "group_acc": group_acc,
        "target_g": target_g,
        "r_ans_g": r_ans_g,
        "r_conf_g": r_conf_g,
        "adv_ans_g": adv_ans_g,
        "adv_conf_g": adv_conf_g,
        "sw_w_g": sw_w_g,
        "mse_target_g": mg * acc_g + (1.0 - mg) * group_acc,
    }
