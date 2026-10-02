"""Unit tests for the pieces of the Pangram 4 recipe that must be exact."""

from __future__ import annotations

import random

import numpy as np
import pytest

from ouroboros.data.clauses import heuristic_clauses, split_sentences
from ouroboros.data.dataset import char_spans_to_token_labels, window_bounds, window_targets
from ouroboros.data.documents import Document, Span
from ouroboros.data.humanize import EVASION_ATTACKS, INCIDENTAL_ARTIFACTS, apply_transforms
from ouroboros.data.mirror import longest_common_substring_ratio
from ouroboros.data.soft_ngrams import label_edit
from ouroboros.data.splice import PATTERNS, splice
from ouroboros.infer.crf import decode, forward_backward, potts_transitions, viterbi
from ouroboros.infer.postprocess import _runs, enforce_min_segments, sentence_majority_vote
from ouroboros.labels import (
    BUCKET_CENTERS,
    N_BUCKETS,
    Provenance,
    fraction_to_bucket,
    segment_prior,
    soft_bucket_target,
    triangular_basis,
    weighted_ai_fraction,
)


# ----------------------------------------------------------------- labels
def test_weighted_ai_fraction_matches_report_formula():
    # f_AI = (0.5*C_AA + C_AG) / (C_H + C_AA + C_AG)
    assert weighted_ai_fraction(100, 0, 0) == 0.0
    assert weighted_ai_fraction(0, 0, 100) == 1.0
    assert weighted_ai_fraction(0, 100, 0) == 0.5
    assert weighted_ai_fraction(50, 20, 30) == pytest.approx((0.5 * 20 + 30) / 100)


def test_triangular_basis_equations():
    assert triangular_basis(0.0).tolist() == [1.0, 0.0, 0.0]
    assert triangular_basis(0.5).tolist() == [0.0, 1.0, 0.0]
    assert triangular_basis(1.0).tolist() == [0.0, 0.0, 1.0]


def test_segment_prior_peaks_at_the_right_buckets():
    """"Peaks at human for bucket 0, ai-assisted mid, ai-generated at 14"."""
    eye = np.eye(N_BUCKETS)
    assert segment_prior(eye[0]).argmax() == Provenance.HUMAN
    assert segment_prior(eye[N_BUCKETS // 2]).argmax() == Provenance.ASSISTED
    assert segment_prior(eye[N_BUCKETS - 1]).argmax() == Provenance.AI


def test_soft_bucket_target_is_a_distribution_centred_on_f():
    target = soft_bucket_target(0.5)
    assert target.sum() == pytest.approx(1.0)
    assert float(target @ BUCKET_CENTERS) == pytest.approx(0.5, abs=1e-6)
    assert fraction_to_bucket(0.0) == 0 and fraction_to_bucket(1.0) == N_BUCKETS - 1


# ----------------------------------------------------------------- windows
def test_window_bounds_stride_and_end_anchor():
    bounds = window_bounds(1200, 512, 256)
    assert bounds[0] == (0, 512)
    assert bounds[-1] == (1200 - 512, 1200)  # final window anchored to the end
    assert all(hi - lo == 512 for lo, hi in bounds)
    assert window_bounds(100, 512, 256) == [(0, 100)]


def test_window_targets_flags_mixed_above_threshold():
    labels = np.array([0] * 90 + [2] * 10)
    char_lens = np.ones(100, dtype=np.int32)
    _, mixed, f_ai = window_targets(labels, char_lens, 0.15, 1.0)
    assert mixed == 0 and f_ai == pytest.approx(0.10)
    labels = np.array([0] * 70 + [2] * 30)
    _, mixed, f_ai = window_targets(labels, char_lens, 0.15, 1.0)
    assert mixed == 1 and f_ai == pytest.approx(0.30)


def test_char_spans_map_onto_tokens_by_midpoint():
    doc = Document("d", "abcdefgh", [Span(0, 4, 0), Span(4, 8, 2)])
    labels, lens = char_spans_to_token_labels([(0, 2), (2, 4), (4, 6), (6, 8)], doc)
    assert labels.tolist() == [0, 0, 2, 2]
    assert lens.tolist() == [2, 2, 2, 2]


# ----------------------------------------------------------------- data
def test_clause_split_is_offset_exact():
    text = "Lucy's place is great, because the mi is warm. I went twice; it never fails."
    clauses = heuristic_clauses(text)
    assert "".join(c.text for c in clauses) == text
    assert all(text[c.start : c.end] == c.text for c in clauses)
    assert "".join(s.text for s in split_sentences(text)) == text


def test_soft_ngrams_recovers_the_three_provenance_classes():
    source = "The bread arrives warm. I have been many times and never had a bad meal."
    target = (
        "The bread arrives warm. I have visited on countless occasions and was never "
        "once disappointed. The patio out back is my favourite spot in summer."
    )
    labeled, f_ai = label_edit(source, target)
    labels = {lc.label for lc in labeled}
    assert int(Provenance.HUMAN) in labels  # verbatim clause survives
    assert int(Provenance.AI) in labels  # novel clause is open generation
    assert 0.0 < f_ai < 1.0


def test_soft_ngrams_is_invariant_to_clause_rearrangement():
    """Reordering without rewriting must stay human (report Section 3.5)."""
    a = "The soup was cold. The bread was warm. The bill was high."
    b = "The bill was high. The soup was cold. The bread was warm."
    _, f_ai = label_edit(a, b)
    assert f_ai == pytest.approx(0.0)


def test_verbatim_overlap_guard():
    assert longest_common_substring_ratio("a b c d e", "a b c d e") == 1.0
    assert longest_common_substring_ratio("a b c d e", "x y z w q") == 0.0


@pytest.mark.parametrize("pattern", PATTERNS)
def test_splice_produces_a_valid_partition(pattern):
    human = "I went there Tuesday. The line was short. My friend hated it. We tipped anyway."
    ai = "The venue offers a comprehensive experience. Guests praise the service. The decor is curated."
    doc = splice(human, ai, random.Random(7), pattern)
    assert doc is not None
    doc.validate()
    assert {s.label for s in doc.spans} == {int(Provenance.HUMAN), int(Provenance.AI)}


@pytest.mark.parametrize("attack", sorted(EVASION_ATTACKS) + sorted(INCIDENTAL_ARTIFACTS))
def test_transforms_keep_spans_aligned(attack):
    doc = Document(
        "d",
        "Furthermore, we utilize numerous comprehensive methods. " * 3
        + "I just went and had a look myself. " * 3,
        [Span(0, 165, 2), Span(165, 270, 0)],
    )
    out = apply_transforms(doc, [attack], random.Random(11))
    out.validate()
    assert len(out.spans) == 2
    assert out.spans[-1].end == len(out.text)


# ----------------------------------------------------------------- CRF
def test_potts_transitions_are_zero_on_the_diagonal():
    trans = potts_transitions(np.array([-10.0, 0.0]), lam=5.0, gamma=0.5)
    assert np.allclose(np.diagonal(trans, axis1=1, axis2=2), 0.0)
    assert (trans[:, 0, 1] <= 0).all()


def test_mixed_evidence_relaxes_the_smoothness_penalty():
    low = potts_transitions(np.array([-10.0]), lam=5.0, gamma=1.0)[0, 0, 1]
    high = potts_transitions(np.array([10.0]), lam=5.0, gamma=1.0)[0, 0, 1]
    assert high > low  # strong mixed evidence makes switching cheaper
    assert high == 0.0


def test_viterbi_smooths_isolated_flips():
    unary = np.zeros((60, 3))
    unary[:30, 0] = 3.0
    unary[30:, 2] = 3.0
    unary[15] = [0.0, 0.0, 3.5]  # a single spurious AI token
    path, _ = decode(unary, np.full(59, -20.0), lam=6.0, gamma=0.5)
    assert _runs(path) == [(0, 30, 0), (30, 60, 2)]


def test_forward_backward_marginals_are_normalised():
    rng = np.random.default_rng(0)
    unary = rng.normal(size=(25, 3))
    trans = potts_transitions(rng.normal(size=24), lam=3.0, gamma=0.5)
    marginals = forward_backward(unary, trans)
    assert np.allclose(marginals.sum(axis=1), 1.0)
    # With no transition penalty the marginals must match the per-token softmax.
    flat = forward_backward(unary, np.zeros((24, 3, 3)))
    expected = np.exp(unary) / np.exp(unary).sum(axis=1, keepdims=True)
    assert np.allclose(flat, expected)


def test_viterbi_matches_brute_force_on_a_short_chain():
    rng = np.random.default_rng(3)
    unary = rng.normal(size=(6, 3))
    trans = potts_transitions(rng.normal(size=5), lam=2.0, gamma=0.5)
    best, best_score = None, -np.inf
    for code in range(3**6):
        path = [(code // 3**i) % 3 for i in range(6)]
        score = sum(unary[i, path[i]] for i in range(6))
        score += sum(trans[i, path[i], path[i + 1]] for i in range(5))
        if score > best_score:
            best, best_score = path, score
    assert viterbi(unary, trans).tolist() == best


# ----------------------------------------------------------------- decoding
def test_sentence_majority_vote_is_sentence_atomic():
    text = "I wrote this first sentence. A model wrote this second one entirely."
    offsets = []
    cursor = 0
    for word in text.split(" "):
        offsets.append((cursor, cursor + len(word)))
        cursor += len(word) + 1
    labels = np.array([0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0])
    voted = sentence_majority_vote(labels, offsets, text)
    assert len(set(voted[:5].tolist())) == 1
    assert len(set(voted[5:].tolist())) == 1


def test_min_segment_merging_removes_short_runs():
    labels = np.array([0] * 40 + [2] * 5 + [0] * 40)
    assert _runs(enforce_min_segments(labels, 32)) == [(0, 85, 0)]
    labels = np.array([0] * 40 + [2] * 40)
    assert len(_runs(enforce_min_segments(labels, 32))) == 2


def test_extract_docx_and_reflow(tmp_path):
    import zipfile

    from ouroboros.infer.extract import docx_text, reflow

    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    xml = (
        f'<w:document {ns}><w:body>'
        '<w:p><w:r><w:t>Premier paragraphe,</w:t></w:r><w:r><w:t xml:space="preserve"> en deux runs.</w:t></w:r></w:p>'
        '<w:p></w:p>'
        '<w:p><w:r><w:t>Second</w:t><w:tab/><w:t>paragraphe.</w:t></w:r></w:p>'
        '</w:body></w:document>'
    )
    path = tmp_path / "doc.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    assert docx_text(path) == "Premier paragraphe, en deux runs.\n\nSecond paragraphe."

    wrapped = "Une ligne coupée\n   au milieu, et celui-\nlà.\n\n  Autre   para.\n 1 / 3 \n"
    assert reflow(wrapped) == "Une ligne coupée au milieu, et celui-là.\n\nAutre para."


def test_normalize_layout_canonical_form():
    from ouroboros.infer.extract import normalize_layout

    clean = "Premier paragraphe.\n\nSecond paragraphe."
    assert normalize_layout(clean) == clean  # already canonical: unchanged

    wrapped = (
        "Vous cherchez à comprendre un phénomène précis : à quel moment un joueur qui rencontre\n"
        "un bug décide de l'exploiter, ou de le signaler aux développeurs du studio concerné.\n"
        "Nous avons donc concentré le projet sur le système qui entoure le jeu plutôt que sur\n"
        "le jeu lui-même. Un jeu de test sert à produire des signaux.\n"
    )
    out = normalize_layout(wrapped)
    assert "\n" not in out.split("\n\n")[0]  # wrapped lines re-joined
    assert "rencontre un bug" in out and "sur le jeu" in out

    assert normalize_layout("- un\n- deux\nSuite du texte.") == "- un\n- deux\n\nSuite du texte."

    flat = " ".join(f"Phrase numéro {i} du texte." for i in range(50))  # 250 words
    assert normalize_layout(flat).count("\n\n") >= 3  # long flat block cut into paragraphs
