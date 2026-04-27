"""
Unit tests for DL engine and chunker.
Run: pytest tests/ -v
"""

import pytest
from core.dl_engine import (
    damerau_levenshtein_distance,
    similarity_score,
    tokenize_text,
    normalize_token,
    search_chunk,
)
from core.chunker import chunk_text_by_words


# ---------------------------------------------------------------------------
# DL distance tests
# ---------------------------------------------------------------------------

class TestDLDistance:
    def test_equal_strings(self):
        assert damerau_levenshtein_distance("abc", "abc") == 0

    def test_insertion(self):
        assert damerau_levenshtein_distance("abc", "abcd") == 1

    def test_deletion(self):
        assert damerau_levenshtein_distance("abcd", "abc") == 1

    def test_substitution(self):
        assert damerau_levenshtein_distance("abc", "axc") == 1

    def test_transposition(self):
        # True DL: transposition counts as 1
        assert damerau_levenshtein_distance("ab", "ba") == 1

    def test_transposition_osa_difference(self):
        # "ca" → "abc": DL = 2, OSA = 3
        assert damerau_levenshtein_distance("ca", "abc") == 2

    def test_empty_strings(self):
        assert damerau_levenshtein_distance("", "") == 0
        assert damerau_levenshtein_distance("abc", "") == 3
        assert damerau_levenshtein_distance("", "abc") == 3

    def test_max_dist_early_termination(self):
        result = damerau_levenshtein_distance("completely", "different", max_dist=2)
        assert result == 3   # exceeds max_dist, returns max_dist+1

    def test_romanian_words(self):
        # "informatica" vs "informatica" with one typo
        assert damerau_levenshtein_distance("informatica", "informaticа") <= 2
        assert damerau_levenshtein_distance("cautare", "cautate") == 1
        assert damerau_levenshtein_distance("algoritm", "algortim") == 1   # transposition


class TestSimilarity:
    def test_perfect_match(self):
        assert similarity_score(0, 10) == 1.0

    def test_zero_query(self):
        assert similarity_score(0, 0) == 1.0

    def test_proportional(self):
        s = similarity_score(2, 10)
        assert abs(s - 0.8) < 1e-9

    def test_clamped_at_zero(self):
        assert similarity_score(20, 5) == 0.0


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_basic(self):
        tokens = tokenize_text("Ana are mere")
        assert [t for _, t in tokens] == ["Ana", "are", "mere"]

    def test_positions(self):
        tokens = tokenize_text("Ana are mere")
        assert tokens[0][0] == 0
        assert tokens[1][0] == 4
        assert tokens[2][0] == 8

    def test_normalize(self):
        assert normalize_token("Mere!") == "mere"
        assert normalize_token("câine,") == "câine"


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_single_chunk(self):
        text = "Ana are mere si pere"
        chunks = list(chunk_text_by_words(text, chunk_size=1000, overlap_size=0))
        assert len(chunks) == 1
        assert chunks[0][1] == text

    def test_multiple_chunks(self):
        text = " ".join(["word"] * 100)
        chunks = list(chunk_text_by_words(text, chunk_size=50, overlap_size=0))
        assert len(chunks) > 1

    def test_overlap_offsets(self):
        text = " ".join([f"w{i:04d}" for i in range(200)])
        chunks = list(chunk_text_by_words(text, chunk_size=100, overlap_size=20))
        # Each chunk except first should start before previous chunk's end
        for i in range(1, len(chunks)):
            assert chunks[i][0] < chunks[i-1][0] + 100

    def test_covers_all_text(self):
        """Union of all chunks must cover the whole text."""
        text = "Ana are mere si pere, iar Ion are pere si gutui."
        chunks = list(chunk_text_by_words(text, chunk_size=20, overlap_size=5))
        # Every character position must appear in at least one chunk
        covered = set()
        for offset, chunk in chunks:
            for i in range(len(chunk)):
                covered.add(offset + i)
        all_positions = set(range(len(text)))
        assert all_positions.issubset(covered)


# ---------------------------------------------------------------------------
# Search chunk tests (integration, no Celery)
# ---------------------------------------------------------------------------

class TestSearchChunk:
    def test_exact_match(self):
        text = "The quick brown fox jumps over the lazy dog"
        matches = search_chunk(text, 0, 0, "fox", max_distance=0)
        assert len(matches) == 1
        assert matches[0].position == text.index("fox")

    def test_fuzzy_match_one_edit(self):
        text = "The quick brownn fox jumps"
        matches = search_chunk(text, 0, 0, "brown", max_distance=1)
        assert len(matches) >= 1
        assert any(m.distance == 1 for m in matches)

    def test_no_match_beyond_distance(self):
        text = "The quick brown fox"
        matches = search_chunk(text, 0, 0, "elephant", max_distance=1)
        assert len(matches) == 0

    def test_phrase_search(self):
        text = "Ana are mere si pere in cosul ei"
        matches = search_chunk(text, 0, 0, "mere si pere", max_distance=0)
        assert len(matches) >= 1

    def test_transposition_match(self):
        text = "The algroithm is efficient"
        matches = search_chunk(text, 0, 0, "algorithm", max_distance=2)
        assert len(matches) >= 1

    def test_chunk_offset_applied(self):
        text = "hello world"
        offset = 5000
        matches = search_chunk(text, offset, 0, "world", max_distance=0)
        assert matches[0].position == 5000 + text.index("world")

    def test_context_extracted(self):
        text = "context before TARGET context after"
        matches = search_chunk(text, 0, 0, "TARGET", max_distance=0)
        assert "context" in matches[0].context

    def test_similarity_score_attached(self):
        text = "informatica distribuita"
        matches = search_chunk(text, 0, 0, "informatica", max_distance=0)
        assert matches[0].similarity == 1.0

    def test_romanian_text(self):
        text = "Algoritmul de cautare fuzzy foloseste distanta Damerau-Levenshtein"
        matches = search_chunk(text, 0, 0, "cautare", max_distance=1)
        assert len(matches) >= 1
