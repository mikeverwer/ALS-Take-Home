"""
test_detector.py - unit tests for the rule-based half of detect().

Runs everywhere with use_embeddings=False: no model to load, no network,
no dependency on sentence-transformers being installed at test time. For
this module that's the whole story on "offline" - detect(use_embeddings=
False) is a pure function of its input string, so there's no I/O to fake,
just the example post strings below acting as the fixture data.
"""
from detector import detect


def test_dollar_prefixed_ticker():
    result = detect("Just bought more $TSLA today", use_embeddings=False)
    assert result.is_stock_related
    assert result.tickers == ["TSLA"]


def test_bare_safe_ticker_matches():
    # TSLA is >=3 chars and not on the common-word-collision list, so a
    # bare mention (no $ prefix, no alias) should still match.
    result = detect("TSLA delivery numbers just dropped, higher than expected.", use_embeddings=False)
    assert result.is_stock_related
    assert "TSLA" in result.tickers


def test_bare_excluded_ticker_does_not_match():
    # BA is deliberately excluded from bare matching - known, documented
    # trade-off (see eval.py's error analysis: posts #19 / #45).
    result = detect("BA still dealing with production delays apparently.", use_embeddings=False)
    assert result.tickers == []


def test_bare_uppercase_common_words_are_never_tickers():
    # The core ambiguity case the whole bare-matching design is built
    # around: short common words in all caps must never fire.
    for text in ["IT was a total disaster from start to finish.",
                 "ALL of my friends are coming to the party this weekend.",
                 "ON second thought, maybe I should reconsider the plan."]:
        result = detect(text, use_embeddings=False)
        assert not result.is_stock_related, f"false positive on: {text!r}"


def test_company_alias_match():
    result = detect("Tesla just opened a new factory overseas.", use_embeddings=False)
    assert result.is_stock_related
    assert "TSLA" in result.tickers


def test_multiple_tickers_in_one_post():
    result = detect("Comparing $TSLA and $F right now, hard to decide.", use_embeddings=False)
    assert set(result.tickers) == {"TSLA", "F"}


def test_fruit_guard_suppresses_apple_the_fruit():
    result = detect("Had an apple with my lunch today, so refreshing.", use_embeddings=False)
    assert not result.is_stock_related


def test_fruit_guard_does_not_over_suppress():
    # Regression test for the substring bug: "patent" contains "ate" and
    # must NOT be treated as fruit-context.
    result = detect("Apple Inc. is facing another lawsuit over patent claims.", use_embeddings=False)
    assert result.is_stock_related
    assert "AAPL" in result.tickers


def test_visa_guard_suppresses_immigration_context():
    result = detect("Visa Inc. denied my green card renewal today.", use_embeddings=False)
    assert not result.is_stock_related


def test_visa_guard_does_not_over_suppress():
    # Regression test: "Indianapolis" contains "india" and must not
    # falsely trigger the country-name heuristic.
    result = detect("Visa Inc. stock jumped today, despite my trip to Indianapolis.", use_embeddings=False)
    assert result.is_stock_related
    assert "V" in result.tickers


def test_chevron_pattern_guard_suppresses_non_company_use():
    result = detect("That shirt has a really nice chevron pattern on it.", use_embeddings=False)
    assert not result.is_stock_related


def test_unrelated_post_has_no_tickers():
    result = detect("Beautiful sunset over the coast tonight, wish everyone could see it.", use_embeddings=False)
    assert not result.is_stock_related
    assert result.tickers == []