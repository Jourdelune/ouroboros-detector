"""Format augmentation of a span-annotated corpus (used for the released model, v7).

Detectors fall for format shortcuts: hard line wraps only ever appeared on human text (PDF / e-mail / Gutenberg
typesetting), everything on one line read as human, paragraph breaks as AI, and short passages (50-160 words) were rare,
so the model fell back to its human prior on them. Each fix is applied to BOTH classes so that no format becomes evidence
for either of them:

    wrap   re-wrap at 70-120 columns        (spaces -> newlines: same length, so character spans stay valid)
    flat   newlines -> spaces               (same length, spans stay valid)
    crop   random 50-160 word excerpt of a pure-label document

It also drops human documents from sources whose "human" label is unreliable (machine translation, pasted chat answers).

usage: python scripts/augment_formats.py --in data/corpus/train.jsonl --out data/corpus/train_aug.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re

NOISY_HUMAN_SOURCES = ("french_instruct", "aya/")


def pure_label(doc: dict):
    labels = {span[2] for span in doc["spans"]}
    return labels.pop() if len(labels) == 1 else None


def is_noisy_human(doc: dict) -> bool:
    domain = str(doc.get("meta", {}).get("domain", ""))
    return any(domain.startswith(n) for n in NOISY_HUMAN_SOURCES) and all(span[2] == 0 for span in doc["spans"])


def wrap(text: str, width: int) -> str:
    out, col = list(text), 0
    for i, ch in enumerate(text):
        col = 0 if ch == "\n" else col + 1
        if ch == " " and col >= width:
            out[i] = "\n"
            col = 0
    return "".join(out)


def crop(doc: dict, label: int, rng: random.Random):
    words = list(re.finditer(r"\S+", doc["text"]))
    n = rng.randint(50, 160)
    if len(words) <= n + 5:
        return None
    i = rng.randint(0, len(words) - n)
    a, b = words[i].start(), words[i + n - 1].end()
    text = doc["text"][a:b]
    return {**doc, "id": doc["id"] + f"-crop{a}", "text": text, "spans": [[0, len(text), label]],
            "meta": {**doc.get("meta", {}), "augment": "crop"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrap", type=int, default=20000, help="wrapped copies per class")
    ap.add_argument("--flat", type=int, default=20000, help="flattened copies per class")
    ap.add_argument("--crop", type=int, default=30000, help="short excerpts per class")
    ap.add_argument("--pool", type=int, default=400000, help="documents sampled to draw augmentations from")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.inp) as f:
        base = [d for d in (json.loads(l) for l in f if l.strip()) if not is_noisy_human(d)]
    pools = {0: [], 2: []}
    for d in rng.sample(base, min(len(base), args.pool)):
        lab = pure_label(d)
        if lab in pools:
            pools[lab].append(d)

    extra = []
    for lab in (0, 2):
        docs = pools[lab]
        rng.shuffle(docs)
        for d in docs[: args.wrap]:
            extra.append({**d, "id": d["id"] + "-wrap", "text": wrap(d["text"], rng.randint(70, 120)),
                          "meta": {**d.get("meta", {}), "augment": "wrap"}})
        for d in docs[args.wrap + 30000 : args.wrap + 30000 + args.flat]:  # disjoint from the cropped ones
            extra.append({**d, "id": d["id"] + "-flat", "text": d["text"].replace("\n", " "),
                          "meta": {**d.get("meta", {}), "augment": "flat"}})
        for d in docs[args.wrap : args.wrap + args.crop]:
            c = crop(d, lab, rng)
            if c:
                extra.append(c)

    out = base + extra
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps({"kept_after_noisy_filter": len(base), "augmented_added": len(extra), "written": len(out)}))


if __name__ == "__main__":
    main()
