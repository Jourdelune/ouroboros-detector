"""Splits must never share content, or every reported metric is inflated."""

from __future__ import annotations

import random

from ouroboros.data.build import _content_key, _splitter, BuildSpec


def test_content_key_ignores_whitespace_and_case():
    a = "The  bread   arrives WARM.\n\nI have been many times."
    b = "the bread arrives warm. I have been many times."
    assert _content_key(a) == _content_key(b)


def test_content_key_separates_different_documents():
    assert _content_key("The bread arrives warm.") != _content_key("The soup was cold.")


def test_identical_content_always_lands_in_the_same_split():
    """The leak we hit: the same Wikipedia article arriving from two recipes.

    Keyed on a positional index those copies get different keys and can fall on
    opposite sides of the train/test boundary; keyed on content they cannot.
    """
    spec = BuildSpec(raid_csv="x")
    text = "Antoine Meillet was a French linguist who studied Indo-European languages."
    assert _splitter(_content_key(text), spec) == _splitter(_content_key(text + "  "), spec)


def test_splitter_is_deterministic_and_covers_all_splits():
    spec = BuildSpec(raid_csv="x")
    keys = [f"src-{i}" for i in range(5000)]
    first = [_splitter(k, spec) for k in keys]
    second = [_splitter(k, spec) for k in keys]
    assert first == second
    assert set(first) == {"train", "eval", "calibration", "test"}


def test_domain_resampling_approaches_the_target_mix():
    """Figure 2 mix: only trims over-represented categories, never duplicates."""
    from dataclasses import dataclass

    from ouroboros.data.build import PAPER_DOMAIN_MIX, resample_to_domain_mix

    @dataclass
    class S:
        domain: str

    # Wildly skewed towards reference, the shape our raw pool actually has.
    samples = [S("wikipedia_fr/reference")] * 50000 + [S("cosmo_stories/x")] * 3000
    samples += [S("industry_finance/x")] * 900 + [S("cosmo_stanford/x")] * 2500
    out = resample_to_domain_mix(samples, PAPER_DOMAIN_MIX, random.Random(0))

    from ouroboros.data.build import domain_of

    counts = {}
    for s in out:
        counts[domain_of(s.domain)] = counts.get(domain_of(s.domain), 0) + 1
    total = sum(counts.values())
    assert total <= len(samples)
    # reference went from 92% of the pool to roughly its 15.9% target
    assert counts["reference"] / total < 0.30
    assert counts["creative"] / total > 0.15


def test_domain_mix_min_keep_bounds_the_loss():
    import random
    from dataclasses import dataclass

    from ouroboros.data.build import PAPER_DOMAIN_MIX, domain_of, resample_to_domain_mix

    @dataclass
    class S:
        domain: str

    # one category almost empty: strict matching would collapse the pool
    samples = [S("wikipedia_fr/reference")] * 50000 + [S("cosmo_stories/x")] * 20000
    samples += [S("industry_finance/x")] * 60
    strict = resample_to_domain_mix(samples, PAPER_DOMAIN_MIX, random.Random(0))
    bounded = resample_to_domain_mix(samples, PAPER_DOMAIN_MIX, random.Random(0), min_keep=0.5)
    assert len(strict) < 0.1 * len(samples)
    assert len(bounded) >= 0.5 * len(samples)
    ref = sum(domain_of(s.domain) == "reference" for s in bounded) / len(bounded)
    assert ref < 50000 / len(samples)  # over-represented category still trimmed
