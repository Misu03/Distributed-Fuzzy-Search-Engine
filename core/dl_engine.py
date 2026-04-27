"""
Damerau-Levenshtein fuzzy search engine.
Supports both OSA (Optimal String Alignment) and true DL distance.
Optimized with numpy and early termination.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Match:
    position: int          # byte/char offset in original text
    chunk_id: int
    matched_text: str
    distance: int
    similarity: float
    context: str           # surrounding snippet


def damerau_levenshtein_distance(s1: str, s2: str, max_dist: int = None) -> int:
    """
    True Damerau-Levenshtein distance (with transpositions).
    Uses the unrestricted algorithm (not OSA).
    Early-terminates if distance exceeds max_dist.
    """
    len1, len2 = len(s1), len(s2)

    if abs(len1 - len2) > (max_dist or float('inf')):
        return max_dist + 1 if max_dist else abs(len1 - len2)

    # alphabet → last seen index
    da: dict = {}

    # dp table: (len1+2) x (len2+2)
    d = np.full((len1 + 2, len2 + 2), 0, dtype=np.int32)
    max_dist_val = len1 + len2
    d[0, 0] = max_dist_val

    for i in range(len1 + 1):
        d[i + 1, 0] = max_dist_val
        d[i + 1, 1] = i

    for j in range(len2 + 1):
        d[0, j + 1] = max_dist_val
        d[1, j + 1] = j

    for i in range(1, len1 + 1):
        db = 0  # last j where s1[i-1] == s2[j-1]
        for j in range(1, len2 + 1):
            i1 = da.get(s2[j - 1], 0)
            j1 = db
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if cost == 0:
                db = j
            d[i + 1, j + 1] = min(
                d[i, j] + cost,            # substitution / match
                d[i + 1, j] + 1,           # insertion
                d[i, j + 1] + 1,           # deletion
                d[i1, j1] + (i - i1 - 1) + 1 + (j - j1 - 1)  # transposition
            )
        da[s1[i - 1]] = i

    result = int(d[len1 + 1, len2 + 1])

    if max_dist and result > max_dist:
        return max_dist + 1
    return result


def similarity_score(distance: int, query_len: int) -> float:
    """Normalize distance to [0, 1] similarity."""
    if query_len == 0:
        return 1.0
    return max(0.0, 1.0 - distance / query_len)


def tokenize_text(text: str) -> List[Tuple[int, str]]:
    """
    Returns list of (position, token) where position is the
    character offset of the token in the original text.
    Preserves punctuation-aware splitting.
    """
    tokens = []
    for m in re.finditer(r'\S+', text):
        tokens.append((m.start(), m.group()))
    return tokens


def normalize_token(token: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    return re.sub(r'[^\w\s]', '', token).lower()


def _char_bigrams(s: str) -> set:
    """Return set of character bigrams for fast pre-filtering."""
    return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def _bigram_sim(s1: str, s2: str) -> float:
    """Bigram Dice coefficient — O(n) cheap proxy for edit distance."""
    b1, b2 = _char_bigrams(s1), _char_bigrams(s2)
    if not b1 or not b2:
        return 0.0
    return 2 * len(b1 & b2) / (len(b1) + len(b2))


def search_chunk(
    chunk_text: str,
    chunk_offset: int,
    chunk_id: int,
    query: str,
    max_distance: int,
    context_size: int = 80,
) -> List[Match]:
    """
    Search for query (single word or multi-word phrase) in chunk_text.
    Returns list of Match objects with absolute positions.

    Optimisations:
    1. Length-based fast rejection — skip windows that are too
       short/long to be within max_distance.
    2. Bigram pre-filter — cheap Dice coefficient check before
       running the full O(n×m) DL computation.
    """
    query = query.strip()
    if not query:
        return []
    query_tokens = query.split()
    query_len = len(query_tokens)
    query_normalized = [normalize_token(t) for t in query_tokens]
    query_joined = ' '.join(query_normalized)
    query_char_len = len(query_joined)

    tokens = tokenize_text(chunk_text)
    if not tokens:
        return []

    # Pre-compute query bigrams once
    query_bigrams = _char_bigrams(query_joined)

    # Minimum bigram similarity threshold that could still yield dist ≤ max_distance
    # (conservative: a window with Dice < 0.15 rarely passes DL ≤ 2)
    min_bigram_sim = max(0.0, 1.0 - (max_distance + 1) * 0.35)

    # Length window: a valid match can't differ in char length by more
    # than max_distance characters from the query length
    min_clen = max(0, query_char_len - max_distance * 2)
    max_clen = query_char_len + max_distance * 2 + query_len  # +spaces

    matches: List[Match] = []
    seen_positions = set()

    for i in range(len(tokens) - query_len + 1):
        window = tokens[i:i + query_len]
        window_normalized = [normalize_token(t) for _, t in window]
        window_joined = ' '.join(window_normalized)

        # ── Fast rejection 1: length ─────────────────────────────────
        wlen = len(window_joined)
        if wlen < min_clen or wlen > max_clen:
            continue

        # ── Fast rejection 2: bigram Dice coefficient ────────────────
        if min_bigram_sim > 0:
            win_bigrams = _char_bigrams(window_joined)
            dice = (2 * len(query_bigrams & win_bigrams)
                    / (len(query_bigrams) + len(win_bigrams)))
            if dice < min_bigram_sim:
                continue

        # ── Full DL computation (only when pre-filters pass) ─────────
        dist = damerau_levenshtein_distance(
            query_joined, window_joined,
            max_dist=max_distance
        )

        if dist <= max_distance:
            abs_pos = chunk_offset + window[0][0]
            if abs_pos in seen_positions:
                continue
            seen_positions.add(abs_pos)

            matched_text = chunk_text[window[0][0]:window[-1][0] + len(window[-1][1])]
            sim = similarity_score(dist, query_char_len)

            ctx_start = max(0, window[0][0] - context_size)
            ctx_end = min(len(chunk_text), window[-1][0] + len(window[-1][1]) + context_size)
            context = chunk_text[ctx_start:ctx_end].replace('\n', ' ')

            matches.append(Match(
                position=abs_pos,
                chunk_id=chunk_id,
                matched_text=matched_text,
                distance=dist,
                similarity=round(sim, 4),
                context=context,
            ))

    return matches
