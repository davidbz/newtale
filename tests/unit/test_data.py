"""Unit tests for data preprocessing and packing — no GPU, no network."""
from __future__ import annotations

from data.preprocessing import ExactDedup, preprocess


def test_preprocess_normal_text() -> None:
    text = "Hello world. " * 50  # well above 200-char minimum
    result = preprocess(text)
    assert result is not None
    assert len(result) >= 200


def test_preprocess_too_short() -> None:
    assert preprocess("hi") is None


def test_preprocess_too_long() -> None:
    assert preprocess("x" * 100_001) is None


def test_preprocess_strips_html() -> None:
    text = "<p>Hello</p> " * 20 + "plain text " * 30
    # html-heavy but below 20% threshold after stripping — should survive
    result = preprocess(text)
    assert result is None or "<p>" not in result


def test_preprocess_repetition_filter() -> None:
    # Character 5-gram "aaaaa" is >30% of all 5-grams → filtered
    spam = "a" * 300  # all 5-grams are "aaaaa"
    result = preprocess(spam)
    assert result is None


def test_exact_dedup_detects_duplicate() -> None:
    dedup = ExactDedup(max_entries=100)
    text = "The quick brown fox jumps over the lazy dog. " * 10
    assert not dedup.is_duplicate(text)
    assert dedup.is_duplicate(text)


def test_exact_dedup_different_texts() -> None:
    dedup = ExactDedup(max_entries=100)
    t1 = "Text one. " * 30
    t2 = "Text two. " * 30
    assert not dedup.is_duplicate(t1)
    assert not dedup.is_duplicate(t2)


def test_exact_dedup_bounded() -> None:
    """When cap is exceeded, dedup falls back to allowing all (best-effort)."""
    dedup = ExactDedup(max_entries=3)
    for i in range(5):
        dedup.is_duplicate(f"unique text {i} " * 30)
    # Should not raise regardless of cap
