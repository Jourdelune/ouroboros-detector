#!/usr/bin/env bash
# Build the span-annotated training corpus from PUBLIC sources, then train and run one round of active learning.
#
#   RAID_CSV=/path/to/raid/train.csv  WORK=/path/to/workdir  ./scripts/build_dataset.sh [stage]
#
# stages (run all in order with no argument): scan  fetch  edit  build  augment  train  mine
#
# What this reproduces: the structure of the pipeline that produced the released model and every command in it. What it
# does NOT reproduce bit-for-bit: the released model's corpus grew over several rounds, with some Hugging Face fetches run
# interactively and with extra frontier-model data from a paid API (`api-edit`, optional below), so counts and exact
# contents differ. Sources are listed in the README ("Training data"); check each licence before reuse.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${RAID_CSV:?set RAID_CSV to the train.csv of the RAID benchmark - https://github.com/liamdugan/raid}"
WORK="${WORK:-data}"
mkdir -p "$WORK"
OB="python -m ouroboros.cli"
STAGE="${1:-all}"
want() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# 1. sample RAID once (the CSV is 12 GB) and cache the subset
want scan && $OB scan --raid-csv "$RAID_CSV" --cache-dir "$WORK/cache" \
    --n-human 30000 --n-ai 30000 --n-humanized 9000

# 2. extra human + AI text from public Hugging Face corpora (wide generators, domains, languages)
#    Recipes with different label conventions are handled one by one in src/ouroboros/data/hf_sources.py.
want fetch && $OB hf-fetch --out "$WORK/hf_samples.jsonl" --n-human 7000 --n-ai 5000 \
    --recipes mage,coling,detection_pile,dmitva,cosmopedia,wildchat,ultrachat,openhermes,arena140k,arena100k,french_alpaca,mgt_multi,fineweb_fr,fineweb_edu,imdb,yelp,arxiv

# 3. AI-edited human text (the "homogeneous mixed" class), generated locally on the GPU.
#    Optional, paid: `$OB api-edit ...` with a frontier model through an OpenAI-compatible endpoint adds modern generators.
want edit && $OB edit --raid-csv "$RAID_CSV" --cache-dir "$WORK/cache" \
    --n-human 30000 --n-ai 30000 --n-humanized 9000 \
    --n 6000 --out "$WORK/edited_pairs.jsonl" --client local:Qwen/Qwen2.5-1.5B-Instruct

# 4. assemble the corpus; splits are made by source document, so there is no leakage between train / eval / test
want build && $OB build-data --raid-csv "$RAID_CSV" --cache-dir "$WORK/cache" \
    --n-human 30000 --n-ai 30000 --n-humanized 9000 --n-spliced 20000 \
    --ai-per-human 1.0 --humanized-per-human 0.5 --hf-ai-per-human 1.0 --max-edited-pairs 0 \
    --out-dir "$WORK/corpus" --hf-samples "$WORK/hf_samples.jsonl" --edited-pairs "$WORK/edited_pairs.jsonl"

# 5. format augmentation on BOTH classes (wrap / flatten / short excerpts) + drop unreliable "human" sources
want augment && python scripts/augment_formats.py --in "$WORK/corpus/train.jsonl" --out "$WORK/corpus/train_aug.jsonl" \
    && mv "$WORK/corpus/train_aug.jsonl" "$WORK/corpus/train.jsonl"

# 6. two-stage training (stage 1 is cheap; stage 2 adds the tokenwise and mixed heads with Repeat2)
want train && { $OB train --config configs/stage1.yaml --stage2-config configs/stage2.yaml
                $OB calibrate --run-dir runs/stage2 --shards "$WORK/corpus/calibration.jsonl" --target-fpr 0.005
                $OB eval --run-dir runs/stage2 --shards "$WORK/corpus/test.jsonl" --out runs/stage2/test_report.json; }

# 7. active learning (report Section 4.2): mine what the model still gets wrong on a reserved pool, add it (upweighted
#    x4 in the released run) to the training set, and continue training. The released model went through five such rounds.
want mine && $OB mine --run-dir runs/stage2 --shards "$WORK/corpus/train.jsonl" --limit 5000 --margin 0.3 \
    --out "$WORK/mined_train.jsonl"
