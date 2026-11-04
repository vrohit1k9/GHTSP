"""
Unit tests for Hashtag Tokenizer module.

Test cases:
- #BreakingNews -> ['breaking', 'news']
- #nationalcomingoutday -> ['national', 'coming', 'out', 'day']
- #Oct12Debate -> ['oct', '12', 'debate']
- #VpDebate -> ['vp', 'debate']
- Single unigram fallback when unsegmentable
- Baseline switch (use_tokenizer=False)
"""

import pytest
from hashtag_tokenizer import tokenize_hashtag, load_wikipedia_vocabulary


@pytest.fixture(scope="module")
def wiki_vocab():
    return load_wikipedia_vocabulary()


def test_hashtag_breaking_news(wiki_vocab):
    tokens = tokenize_hashtag("#BreakingNews", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["breaking", "news"]


def test_hashtag_national_coming_out_day(wiki_vocab):
    tokens = tokenize_hashtag("#nationalcomingoutday", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["national", "coming", "out", "day"]


def test_hashtag_oct12_debate(wiki_vocab):
    tokens = tokenize_hashtag("#Oct12Debate", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["oct", "12", "debate"]


def test_hashtag_vp_debate(wiki_vocab):
    tokens = tokenize_hashtag("#VpDebate", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["vp", "debate"]


def test_hashtag_unigram_fallback(wiki_vocab):
    # Non-dictionary gibberish string
    tokens = tokenize_hashtag("#xyzqwkzzp", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["xyzqwkzzp"]


def test_hashtag_baseline_mode(wiki_vocab):
    # When tokenizer is disabled (SEDTWik baseline), raw hashtag is kept as a single token
    tokens = tokenize_hashtag("#BreakingNews", vocab=wiki_vocab, use_tokenizer=False)
    assert tokens == ["breakingnews"]


def test_hashtag_fifa_world_cup(wiki_vocab):
    tokens = tokenize_hashtag("#FIFAWorldCup", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["fifa", "world", "cup"]


def test_hashtag_world_cup_final(wiki_vocab):
    tokens = tokenize_hashtag("#WorldCupFinal", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["world", "cup", "final"]


def test_hashtag_ukraine_war_alert(wiki_vocab):
    tokens = tokenize_hashtag("#UkraineWarAlert", vocab=wiki_vocab, use_tokenizer=True)
    assert tokens == ["ukraine", "war", "alert"]
