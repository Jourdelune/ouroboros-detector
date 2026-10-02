"""HuggingFace corpora that widen the generator and domain distribution.

RAID alone supplies only ~13k distinct human source documents and 2024-era open
generators. The report trains on a wide distribution of domains, languages and
generator models (Section 2), so we pull additional human and machine text from
public detection corpora and from large LLM-output datasets.

Each recipe is responsible for the label convention of its own dataset -- these
are *not* consistent across the Hub (MAGE uses label 1 = human, COLING uses
label 0 = human), so every convention below was verified against the data.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from tqdm.auto import tqdm

HUMAN = "human"
AI = "ai"


@dataclass
class HFSample:
    text: str
    label: str  # HUMAN | AI
    generator: str
    domain: str
    pair_id: str | None = None  # set when a row yields a topic-matched pair


@dataclass(frozen=True)
class HFRecipe:
    key: str
    path: str
    split: str
    extract: Callable[[dict], list[HFSample]]
    config: str | None = None
    note: str = ""


# --------------------------------------------------------------------------
# extractors
# --------------------------------------------------------------------------
def _mage(row: dict) -> list[HFSample]:
    """yaful/MAGE: label 1 = human, 0 = machine; `src` = "<domain>_<origin>"."""
    src = str(row.get("src", ""))
    text = (row.get("text") or "").strip()
    if not text:
        return []
    domain = src.split("_")[0]
    if str(row.get("label")) == "1":
        return [HFSample(text, HUMAN, "human", domain)]
    # e.g. "cmv_machine_continuation_t0_11b" -> generator "t0_11b"
    generator = src.split("machine_", 1)[-1] if "machine_" in src else src
    return [HFSample(text, AI, generator.replace("continuation_", "").replace("specified_", ""), domain)]


def _coling(row: dict) -> list[HFSample]:
    """Jinyan1/COLING_2025_MGT_en: label 0 = human, 1 = machine."""
    text = (row.get("text") or "").strip()
    if not text:
        return []
    domain = str(row.get("sub_source") or row.get("source") or "misc")
    if str(row.get("label")) == "0":
        return [HFSample(text, HUMAN, "human", domain)]
    return [HFSample(text, AI, str(row.get("model") or "unknown"), domain)]


def _detection_pile(row: dict) -> list[HFSample]:
    """artem9k/ai-text-detection-pile: `source` is literally 'human' or 'ai'."""
    text = (row.get("text") or "").strip()
    source = str(row.get("source", "")).lower()
    if not text or source not in (HUMAN, AI):
        return []
    return [HFSample(text, source, "human" if source == HUMAN else "detection-pile-ai", "mixed")]


def _cosmopedia(row: dict) -> list[HFSample]:
    """HuggingFaceTB/cosmopedia: long-form synthetic prose from Mixtral-8x7B."""
    text = (row.get("text") or "").strip()
    if not text:
        return []
    return [HFSample(text, AI, "mixtral-8x7b-instruct", str(row.get("format") or "textbook"))]


def _first_assistant_turn(conversation) -> str:
    if isinstance(conversation, str):
        parsed = None
        for loader in (ast.literal_eval, json.loads):
            try:
                parsed = loader(conversation)
                break
            except Exception:
                continue
        if parsed is None:
            return ""
        conversation = parsed
    if not isinstance(conversation, list):
        return ""
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from")
        if role in ("assistant", "gpt"):
            return str(turn.get("content") or turn.get("value") or "").strip()
    return ""


def _wildchat(row: dict) -> list[HFSample]:
    """allenai/WildChat-1M: real GPT-3.5/GPT-4 replies to real user prompts."""
    if str(row.get("language")) != "English":
        return []
    text = _first_assistant_turn(row.get("conversation"))
    if not text:
        return []
    return [HFSample(text, AI, str(row.get("model") or "gpt-4"), "chat")]


def _ultrachat(row: dict) -> list[HFSample]:
    text = _first_assistant_turn(row.get("messages"))
    return [HFSample(text, AI, "gpt-3.5-turbo", "chat")] if text else []


def _openhermes(row: dict) -> list[HFSample]:
    text = _first_assistant_turn(row.get("conversations"))
    if not text:
        return []
    return [HFSample(text, AI, str(row.get("model") or "gpt-4"), str(row.get("category") or "chat"))]


def _dmitva(row: dict) -> list[HFSample]:
    """dmitva/human_ai_generated_text: a topic-matched human/AI pair per row.

    These pairs are exactly what synthetic mirroring produces, so they also feed
    the heterogeneous splicer.
    """
    human = (row.get("human_text") or "").strip()
    ai = (row.get("ai_text") or "").strip()
    pair_id = str(row.get("id") or "")
    out: list[HFSample] = []
    if human:
        out.append(HFSample(human, HUMAN, "human", "essay", pair_id))
    if ai:
        out.append(HFSample(ai, AI, "dmitva-ai", "essay", pair_id))
    return out


# --------------------------------------------------------------------------
# scale + multilingual extractors
# --------------------------------------------------------------------------
#: WildChat covers many languages; we keep the ones worth training on rather
#: than the long tail, and French because it is a first-class target here.
WILDCHAT_LANGS = {
    "English", "French", "Spanish", "German", "Portuguese", "Italian",
    "Dutch", "Polish", "Russian", "Japanese", "Chinese", "Korean", "Turkish",
}


def _wildchat_multi(row: dict) -> list[HFSample]:
    """allenai/WildChat-4.8M: real ChatGPT replies, many languages.

    This is the single best source of genuine ChatGPT prose: real user prompts,
    real gpt-3.5/gpt-4 answers, and a language label we can filter on.
    """
    lang = str(row.get("language") or "")
    if lang not in WILDCHAT_LANGS:
        return []
    text = _first_assistant_turn(row.get("conversation"))
    if not text:
        return []
    model = str(row.get("model") or "gpt-4")
    return [HFSample(text, AI, model, f"chat/{lang.lower()}")]


def _claude_evol(row: dict) -> list[HFSample]:
    text = (row.get("output") or "").strip()
    return [HFSample(text, AI, "claude-evol-instruct", "instruct")] if text else []


def _claude_opus_modern(row: dict) -> list[HFSample]:
    """TeichAI Claude Opus 4.7 responses (reasoning traces excluded)."""
    text = (row.get("response") or "").strip()
    model = str(row.get("model") or "claude-opus-4.7")
    return [HFSample(text, AI, model, "instruct")] if text else []


def _claude_messages(row: dict) -> list[HFSample]:
    text = _first_assistant_turn(row.get("messages"))
    return [HFSample(text, AI, "claude-sonnet-4.6/opus-4.8", "instruct")] if text else []


def _french_alpaca(row: dict) -> list[HFSample]:
    """jpacifico/French-Alpaca: French instruction answers generated by GPT."""
    text = (row.get("output") or "").strip()
    return [HFSample(text, AI, "gpt-french-alpaca", "instruct/french")] if text else []


def _french_instruct(row: dict) -> list[HFSample]:
    """angeluriot/french_instruct: carries an `author` field, so it yields both classes."""
    conv = row.get("conversation")
    if isinstance(conv, str):
        try:
            conv = ast.literal_eval(conv)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(conv, list):
        return []
    text = ""
    for turn in conv:
        if isinstance(turn, dict) and turn.get("role") in ("assistant", "bot", "chatbot"):
            text = str(turn.get("text") or turn.get("content") or "").strip()
            break
    if not text:
        return []
    author = str(row.get("author") or "").lower()
    if author in ("human", "humain"):
        return [HFSample(text, HUMAN, "human", "instruct/french")]
    return [HFSample(text, AI, f"french-{author or 'chatbot'}", "instruct/french")]


def _wikipedia(row: dict) -> list[HFSample]:
    text = (row.get("text") or "").strip()
    return [HFSample(text, HUMAN, "human", "reference")] if text else []


def _fineweb(row: dict) -> list[HFSample]:
    text = (row.get("text") or "").strip()
    return [HFSample(text, HUMAN, "human", "web")] if text else []


def _aya(row: dict) -> list[HFSample]:
    """CohereLabs/aya_dataset: human-written answers in many languages."""
    text = (row.get("targets") or "").strip()
    lang = str(row.get("language") or "misc").lower()
    return [HFSample(text, HUMAN, "human", f"aya/{lang}")] if text else []


# --------------------------------------------------------------------------
# 30-language human coverage + AI-paraphrase pair sources
# --------------------------------------------------------------------------
#: Wikipedia language codes used for the human side. Broad coverage matters for
#: the false-positive rate: a detector that has never seen a language will read
#: its unfamiliar surface statistics as "machine".
WIKI_LANGS = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "ja",
    "zh", "ar", "tr", "ko", "sv", "vi", "id", "uk", "fa", "he",
    "cs", "ro", "hu", "da", "fi", "no", "el", "th", "hi", "bn",
]


def _mgt_multi(row: dict) -> list[HFSample]:
    """kanwal-mehreen18 multilingual MGT: original human text + AI-modified text."""
    original = (row.get("Original text") or "").strip()
    modified = (row.get("Modified text") or "").strip()
    iso = str(row.get("ISO") or "xx").lower()
    llm = str(row.get("LLM used") or "unknown")
    kind = str(row.get("Type") or "")
    out: list[HFSample] = []
    if original:
        out.append(HFSample(original, HUMAN, "human", f"mgt/{iso}"))
    if modified and kind == "Rewritten":
        out.append(HFSample(modified, AI, llm, f"mgt/{iso}"))
    return out


# --------------------------------------------------------------------------
# frontier-model outputs (2026 generation)
# --------------------------------------------------------------------------
# The v2 corpus was 94.5% pre-2024 generators (gpt-3.5-turbo-0301, gpt-4-0314,
# mixtral, gpt2). Current frontier models write nothing like those, which is why
# a document written by one read as human. These recipes fix the generator mix.

_CODE_MARKERS = ("```", "def ", "import ", "function(", "const ", "#include", "</", "{}", "});")


def _is_prose(text: str) -> bool:
    """Reject code and tool traces: the report scopes detection to prose."""
    if not text:
        return False
    hits = sum(text.count(m) for m in _CODE_MARKERS)
    if hits >= 3:
        return False
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / max(1, len(text)) > 0.80


def _frontier_messages(model: str, domain: str = "chat"):
    def extract(row: dict) -> list[HFSample]:
        text = _first_assistant_turn(row.get("messages"))
        if not _is_prose(text):
            return []
        name = str(row.get("model") or model)
        return [HFSample(text, AI, name, domain)]

    return extract


def _manus_frontier(row: dict) -> list[HFSample]:
    """Multi-teacher distillation: GPT-5.5, Gemini 3.1 Pro, Grok 4, Fable 5, Mythos 5…"""
    text = (row.get("response") or "").strip()
    if not _is_prose(text):
        return []
    domain = str(row.get("category") or row.get("source") or "mixed")
    return [HFSample(text, AI, "frontier-distill-2026", f"manus/{domain}")]


def _instruction_output(model: str, domain: str = "chat"):
    def extract(row: dict) -> list[HFSample]:
        text = (row.get("output") or "").strip()
        return [HFSample(text, AI, model, domain)] if _is_prose(text) else []

    return extract


def _arena_turns(conversation) -> list[str]:
    """Assistant texts of an LMArena conversation (140k and 100k layouts)."""
    out = []
    for msg in conversation if conversation is not None else []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content.strip())
            continue
        parts = [c.get("text") or "" for c in (content if content is not None else [])
                 if isinstance(c, dict) and c.get("type", "text") == "text"]
        out.append("\n".join(parts).strip())
    return [t for t in out if t]


def _arena(row: dict) -> list[HFSample]:
    """LMArena battles: two frontier answers to the same real user prompt.

    Both sides are AI; prompts are real users, so the topics are what people
    actually ask, in whatever language they ask it.
    """
    if row.get("is_code"):
        return []
    lang = str(row.get("language") or "und")
    out = []
    for side in ("a", "b"):
        model = str(row.get(f"model_{side}") or "arena")
        for text in _arena_turns(row.get(f"conversation_{side}"))[:1]:
            if _is_prose(text):
                out.append(HFSample(text, AI, model, f"arena/{lang}"))
    return out


def _peer_review(row: dict) -> list[HFSample]:
    """IntelLabs peer reviews: label 0 = the real human review, 1 = an LLM
    review of the same paper, so both classes share topic and register."""
    text = str(row.get("review") or "").strip()
    if not text:
        return []
    if int(row.get("label", -1)) == 0:
        return [HFSample(text, HUMAN, "human", "abstracts/peer_review")]
    model = str(row.get("custom_id") or "llm").rsplit("-", 2)[0].split("-ICLR")[0].split("-NeurIPS")[0]
    return [HFSample(text, AI, model, "abstracts/peer_review")]


_RASBT_HUMAN_DOMAIN = {
    "pmc": "abstracts/pmc", "plos": "abstracts/plos", "arxiv": "abstracts/arxiv",
    "arxiv-preprints": "abstracts/arxiv", "openstax": "abstracts/openstax",
    "wikimedia": "wikipedia/en", "gutenberg": "wp/gutenberg", "stackexchange": "reddit/stackexchange",
}


def _rasbt(row: dict) -> list[HFSample]:
    """rasbt/human-vs-ai-50k: label 0 = human chunk (papers, books, wiki, Q&A),
    1 = a 2025-26 model's response on the same material."""
    text = str(row.get("text") or "").strip()
    if not text:
        return []
    coll = str(row.get("text_collection") or "")
    if str(row.get("label")) == "0":
        return [HFSample(text, HUMAN, "human", _RASBT_HUMAN_DOMAIN.get(coll, f"web/{coll}"))]
    if not _is_prose(text):
        return []
    model = str(row.get("model") or row.get("generator") or coll or "llm")
    return [HFSample(text, AI, model, f"rasbt/{coll}")]


def _mild(domain: str):
    """mild-rgb Reddit sets: the human side ships without text (licensing),
    the AI side is 2026 frontier models (GPT-5.6 luna, Gemini 3.7, Grok 4.6...)."""

    def extract(row: dict) -> list[HFSample]:
        text = str(row.get("text") or "").strip()
        if row.get("label") != "ai" or text in ("", "None"):
            return []
        model = str(row.get("generator") or row.get("model") or "llm")
        return [HFSample(text, AI, model, domain)]

    return extract


def _field(field: str, model: str, domain: str):
    def extract(row: dict) -> list[HFSample]:
        text = str(row.get(field) or "").strip()
        return [HFSample(text, AI, model, domain)] if _is_prose(text) else []

    return extract


def _sharechat(platform: str):
    """tucnguyen/ShareChat: real shared conversations, one row per message.

    The `role` field separates genuine human prompts from genuine assistant
    output, and `platform`/`model` name the system that produced it -- so this
    gives current ChatGPT and Claude prose with reliable provenance.
    """

    def extract(row: dict) -> list[HFSample]:
        text = (row.get("plain_text") or "").strip()
        if not text:
            return []
        role = str(row.get("role") or "").lower()
        if role in ("user", "human"):
            # Deliberately dropped. Long user turns are prompts, and very often
            # pasted content -- including AI prose the user is handing back to
            # the model ("combine that story with this one"). Labeling those as
            # human would teach the detector that AI text is human, which is the
            # exact failure we are trying to remove. Human prose comes from
            # Wikipedia, FineWeb, Aya and the detection corpora instead.
            return []
        # ShareChat labels assistant turns "llm"; other exports use other names.
        if role not in ("assistant", "model", "bot", "llm", "ai", "gpt", "chatbot"):
            return []
        model = str(row.get("model") or "").strip()
        if not model or model.lower() in ("none", "default_model_slug", "assistant"):
            model = platform
        return [HFSample(text, AI, f"{platform}:{model}", f"sharechat/{platform}")]

    return extract


def _industry(row: dict) -> list[HFSample]:
    """BAAI IndustryCorpus: human professional / sector writing."""
    text = (row.get("text") or row.get("content") or "").strip()
    return [HFSample(text, HUMAN, "human", "professional")] if text else []


def _arxiv(row: dict) -> list[HFSample]:
    """arXiv abstracts (snapshot up to 2021, before chat LLMs): human scientific prose."""
    text = " ".join(str(row.get("abstract") or "").split())
    return [HFSample(text, HUMAN, "human", "abstracts/arxiv")] if text else []


def _review(name: str):
    """Human product / business / film reviews, all collected before 2020."""

    def extract(row: dict) -> list[HFSample]:
        text = str(row.get("text") or "").replace("<br />", "\n").replace("\\n", "\n").strip()
        return [HFSample(text, HUMAN, "human", f"reviews/{name}")] if text else []

    return extract


#: cosmopedia configs that produce *document*-register AI prose (textbooks,
#: articles, how-tos) rather than chat turns. Register matters independently of
#: model era: a detector trained only on chat answers has never seen what an
#: AI-written document looks like.
COSMOPEDIA_DOC_CONFIGS = [
    "stanford", "openstax", "khanacademy", "wikihow", "stories", "web_samples_v1",
]


RECIPES: dict[str, HFRecipe] = {
    r.key: r
    for r in [
        HFRecipe("mage", "yaful/MAGE", "train", _mage, note="27 generators x 10 domains"),
        HFRecipe("coling", "Jinyan1/COLING_2025_MGT_en", "train", _coling, note="GPT-4/4o, Llama-3"),
        HFRecipe("detection_pile", "artem9k/ai-text-detection-pile", "train", _detection_pile),
        HFRecipe("cosmopedia", "HuggingFaceTB/cosmopedia", "train", _cosmopedia, config="web_samples_v2"),
        *[
            HFRecipe(f"cosmo_{cfg}", "HuggingFaceTB/cosmopedia", "train", _cosmopedia,
                     config=cfg, note="AI prose in document register")
            for cfg in COSMOPEDIA_DOC_CONFIGS
        ],
        *[
            HFRecipe(f"industry_{sector}", f"BAAI/IndustryCorpus_{sector}", "train",
                     _industry, note="human professional writing")
            for sector in ("finance", "law", "news", "education")
        ],
        HFRecipe("arxiv", "gfissore/arxiv-abstracts-2021", "train", _arxiv,
                 note="human scientific abstracts"),
        HFRecipe("yelp", "Yelp/yelp_review_full", "train", _review("yelp")),
        HFRecipe("imdb", "stanfordnlp/imdb", "train", _review("imdb")),
        *[
            HFRecipe(f"amazon_{lang}", f"SetFit/amazon_reviews_multi_{lang}", "train",
                     _review(f"amazon_{lang}"), note="human reviews, 6 languages")
            for lang in ("en", "fr", "de", "es", "ja", "zh")
        ],
        HFRecipe("arena140k", "lmarena-ai/arena-human-preference-140k", "train", _arena,
                 note="2025 frontier answers: Claude 4, o3, GPT-4.1, Gemini 2.5..."),
        HFRecipe("arena100k", "lmarena-ai/arena-human-preference-100k", "train", _arena,
                 note="2024 frontier answers"),
        *[
            HFRecipe(f"peer_{cfg}_{model.lower()}", "IntelLabs/AI-Peer-Review-Detection-Benchmark",
                     model, _peer_review, config=cfg, note="paired human / LLM peer reviews")
            for cfg in ("calibration", "test", "extended")
            for model in ("GPT4o", "Claude_Sonnet_3.5", "Gemini_1.5_Pro", "Llama_3.1_70b", "Qwen_2.5_72b")
            if cfg != "extended" or model in ("GPT4o", "Llama_3.1_70b")
        ],
        *[
            HFRecipe(f"rasbt_{split}", "rasbt/human-vs-ai-50k", split, _rasbt,
                     note="human papers/books/wiki vs 2025-26 model responses")
            for split in ("train", "validation", "test")
        ],
        *[
            HFRecipe(f"mild_{name}_{split}", f"mild-rgb/{name}-human-vs-ai", split,
                     _mild(f"reddit/{name}"), note="GPT-5.6 luna, Gemini 3.7, Grok 4.6...")
            for name in ("aita", "eli5") for split in ("train", "validation", "test")
        ],
        HFRecipe("codeflame_frontier",
                 "CodeFlame/FIXED-Cleaned-Claude-Sonnet-5-Grok-4.5-ChatGPT-5.6-Luna-Qwen-3.8-MAX",
                 "train", _frontier_messages("sonnet5-chatgpt5.6-mix")),
        HFRecipe("opus45_writing", "Crownelius/Opus-4.5-WritingStyle-1000x", "train",
                 _field("text", "claude-opus-4.5", "wp/opus45")),
        HFRecipe("gemini3_creative", "Crownelius/Creative-Writing-Gemini3Pro-2700x", "train",
                 _field("response", "gemini-3-pro", "wp/gemini3")),
        HFRecipe("gpt4o_writing", "Gryphe/ChatGPT-4o-Writing-Prompts", "train",
                 _frontier_messages("chatgpt-4o-latest", "wp/gpt4o")),
        HFRecipe("wildchat", "allenai/WildChat-1M", "train", _wildchat),
        HFRecipe("ultrachat", "HuggingFaceH4/ultrachat_200k", "train_sft", _ultrachat),
        HFRecipe("openhermes", "teknium/OpenHermes-2.5", "train", _openhermes),
        HFRecipe("dmitva", "dmitva/human_ai_generated_text", "train", _dmitva),
        # --- scale + ChatGPT / Claude focus + French ------------------------
        HFRecipe("wildchat48", "allenai/WildChat-4.8M", "train", _wildchat_multi,
                 note="real ChatGPT, multilingual incl. French"),
        HFRecipe("claude_evol", "Norquinal/claude_evol_instruct_210k", "train", _claude_evol,
                 note="210k Claude outputs"),
        HFRecipe("claude_opus", "TeichAI/lordx64-claude-opus-4.7-max-cleaned", "train",
                 _claude_opus_modern, note="Claude Opus 4.7"),
        HFRecipe("claude_modern",
                 "thetrillioniar/claude-sonnet-4.6-opus-4.8-mythos-5-fable-5-openai-finetuning-dataset",
                 "train", _claude_messages, note="Claude Sonnet 4.6 / Opus 4.8"),
        HFRecipe("french_alpaca", "jpacifico/French-Alpaca-dataset-Instruct-110K", "train",
                 _french_alpaca, note="French AI text (GPT-generated)"),
        HFRecipe("french_instruct", "angeluriot/french_instruct", "train", _french_instruct,
                 note="French, both classes"),
        # --- real shared conversations (ChatGPT, Claude, Gemini, Grok) -------
        *[
            HFRecipe(f"sharechat_{plat}", "tucnguyen/ShareChat", "train",
                     _sharechat(plat), config=plat,
                     note=f"real shared {plat} conversations")
            for plat in ("chatgpt", "claude", "gemini", "grok", "perplexity")
        ],
        # --- frontier generation (2026) -------------------------------------
        HFRecipe("manus_frontier",
                 "Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset",
                 "train", _manus_frontier, note="GPT-5.5, Gemini 3.1 Pro, Grok 4, Fable 5, Mythos 5"),
        HFRecipe("fable5_premium", "saidutta69/fable-5-premium", "train",
                 _frontier_messages("claude-fable-5"), note="Claude Fable 5"),
        HFRecipe("fable5_premium2", "saidutta69/fable-5-premium-v2", "train",
                 _frontier_messages("claude-fable-5")),
        HFRecipe("fable51_premium", "saidutta69/fable-5.1-premium", "train",
                 _frontier_messages("claude-fable-5.1")),
        HFRecipe("fable5_chat", "armand0e/Fable-5-Chat", "train",
                 _frontier_messages("claude-fable-5")),
        HFRecipe("fable51_reasoning", "MoreThought/Fable-5.1-Max-Reasoning-Filtered-5000x", "train",
                 _frontier_messages("claude-fable-5.1")),
        HFRecipe("sonnet5", "mondk/claude-sonnet5-jsonl", "train",
                 _instruction_output("claude-sonnet-5"), note="Claude Sonnet 5"),
        HFRecipe("opus48_thinking", "11-47/claude_opus_4.8_max_thinking_5k_v2", "train",
                 _frontier_messages("claude-opus-4.8")),
        HFRecipe("opus48_distill", "11-47/claude_opus_4.8_distill_5k", "train",
                 _frontier_messages("claude-opus-4.8")),
        HFRecipe("gpt56_luna", "Roman1111111/GPT-5.6-luna-reasoning-102881x", "train",
                 _frontier_messages("gpt-5.6-luna"), note="GPT-5.6 Luna, 102k rows"),
        HFRecipe("opus5_mmlu", "learning-machine-inc/mmlu-pro-cot-opus5", "train",
                 _frontier_messages("claude-opus-5")),
        HFRecipe("opus5_musr", "learning-machine-inc/musr-cot-opus5", "train",
                 _frontier_messages("claude-opus-5")),
        HFRecipe("opus5_bbh", "learning-machine-inc/bbh-cot-opus5", "train",
                 _frontier_messages("claude-opus-5")),
        HFRecipe("mgt_multi", "kanwal-mehreen18/Multilingual_Machine_Generated_Text_Detection",
                 "train", _mgt_multi, note="multilingual human + AI-modified"),
        *[
            HFRecipe(f"wikipedia_{code}", "wikimedia/wikipedia", "train", _wikipedia,
                     config=f"20231101.{code}", note=f"human reference text ({code})")
            for code in WIKI_LANGS
        ],
        HFRecipe("fineweb_fr", "HuggingFaceFW/fineweb-2", "train", _fineweb, config="fra_Latn",
                 note="French general web"),
        HFRecipe("fineweb_edu", "HuggingFaceFW/fineweb-edu", "train", _fineweb,
                 config="sample-10BT", note="English educational web, human"),
        HFRecipe("aya", "CohereLabs/aya_dataset", "train", _aya, note="human, many languages"),
    ]
}


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
class _Reservoir:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def offer(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = item

    @property
    def full(self) -> bool:
        return len(self.items) >= self.capacity


def sample_recipe(
    recipe: HFRecipe,
    n_human: int,
    n_ai: int,
    min_chars: int = 500,
    max_chars: int = 12000,
    max_rows: int = 400_000,
    seed: int = 0,
    progress: bool = True,
) -> list[HFSample]:
    """Stream a dataset once and reservoir-sample it per class.

    Plain streaming plus reservoirs, rather than ``.shuffle(buffer_size=...)``:
    the reservoir is a genuinely uniform sample over everything seen, and it
    avoids the shuffle buffer's background threads.
    """
    from datasets import load_dataset

    rng = random.Random(f"{seed}:{recipe.key}")
    dataset = load_dataset(recipe.path, recipe.config, split=recipe.split, streaming=True)
    reservoirs = {HUMAN: _Reservoir(n_human, rng), AI: _Reservoir(n_ai, rng)}

    bar = tqdm(desc=f"hf:{recipe.key}", unit="row", total=max_rows, disable=not progress)
    seen = 0
    # Hub datasets whose parquet shards disagree on schema raise mid-stream.
    # Keep whatever was sampled before the failure instead of discarding the
    # whole recipe -- a partial sample is still useful data.
    iterator = iter(dataset)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            print(f"[hf:{recipe.key}] stream ended early after {seen} rows: "
                  f"{type(exc).__name__}: {str(exc)[:90]}")
            break
        seen += 1
        bar.update(1)
        if seen >= max_rows:
            break
        try:
            extracted = recipe.extract(row)
        except Exception:
            continue
        for sample in extracted:
            if not (min_chars <= len(sample.text) <= max_chars):
                continue
            reservoir = reservoirs.get(sample.label)
            if reservoir is not None and reservoir.capacity > 0:
                reservoir.offer(sample)
        if all(r.full or r.capacity == 0 for r in reservoirs.values()) and seen > max_rows // 8:
            break  # both classes are saturated with a well-mixed sample
    bar.close()

    out = reservoirs[HUMAN].items + reservoirs[AI].items
    rng.shuffle(out)
    return out


def write_samples(path: str | Path, samples: Iterable[HFSample], source: str) -> int:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as fh:
        for s in samples:
            fh.write(
                json.dumps(
                    {
                        "text": s.text,
                        "label": s.label,
                        "generator": s.generator,
                        "domain": s.domain,
                        "pair_id": s.pair_id,
                        "recipe": source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def read_samples(path: str | Path) -> list[HFSample]:
    from .documents import iter_json_lines

    out: list[HFSample] = []
    for row in iter_json_lines(path):
        out.append(
            HFSample(
                text=row["text"],
                label=row["label"],
                generator=row["generator"],
                domain=f'{row["recipe"]}/{row["domain"]}',
                pair_id=row.get("pair_id"),
            )
        )
    return out


# --------------------------------------------------------------------------
# (source, AI-rewritten) pair sources
# --------------------------------------------------------------------------
# These feed the Soft N-Grams labeler directly, which is how the `ai-assisted`
# class is built. They are far cheaper than generating edits ourselves, and
# `humarin/chatgpt-paraphrases` in particular is genuine ChatGPT output.


def _pairs_chatgpt_paraphrase(row: dict) -> list[dict]:
    import ast

    source = (row.get("text") or "").strip()
    raw = row.get("paraphrases")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if not source or not isinstance(raw, list) or not raw:
        return []
    target = str(raw[0]).strip()
    if not target or target == source:
        return []
    return [
        {
            "source": source,
            "target": target,
            "generator": "chatgpt-paraphrase",
            "tier": "hf",
            "instruction": "paraphrase",
            "intensity": "medium",
            "source_name": "hf-paraphrase",
        }
    ]


def _pairs_mgt_multi(row: dict) -> list[dict]:
    """Original vs AI-modified text, keeping the language and the LLM."""
    source = (row.get("Original text") or "").strip()
    target = (row.get("Modified text") or "").strip()
    kind = str(row.get("Type") or "")
    if not source or not target or source == target or kind == "Unchanged":
        return []
    return [
        {
            "source": source,
            "target": target,
            "generator": str(row.get("LLM used") or "unknown"),
            "tier": "hf",
            "instruction": kind.lower(),
            "intensity": "heavy" if kind == "Rewritten" else "medium",
            "source_name": f"mgt-{str(row.get('ISO') or 'xx').lower()}",
        }
    ]


PAIR_RECIPES: dict[str, tuple[str, str | None, str, object]] = {
    "chatgpt_paraphrase": ("humarin/chatgpt-paraphrases", None, "train", _pairs_chatgpt_paraphrase),
    "mgt_pairs": (
        "kanwal-mehreen18/Multilingual_Machine_Generated_Text_Detection",
        None,
        "train",
        _pairs_mgt_multi,
    ),
}


def sample_pairs(
    key: str, n: int, min_chars: int = 300, max_rows: int = 400_000, seed: int = 0
) -> list[dict]:
    """Reservoir-sample (source, AI-rewritten) pairs from a Hub dataset."""
    from datasets import load_dataset

    path, config, split, extract = PAIR_RECIPES[key]
    rng = random.Random(f"{seed}:{key}")
    dataset = load_dataset(path, config, split=split, streaming=True)
    reservoir = _Reservoir(n, rng)

    bar = tqdm(desc=f"pairs:{key}", unit="row", total=max_rows)
    for i, row in enumerate(dataset):
        if i >= max_rows:
            break
        bar.update(1)
        for pair in extract(row):
            if len(pair["source"]) >= min_chars and len(pair["target"]) >= min_chars:
                pair["id"] = f"{key}-{i}"
                pair["source_id"] = f"{key}-{i}"
                reservoir.offer(pair)
        if reservoir.full and i > max_rows // 6:
            break
    bar.close()
    return reservoir.items


# --------------------------------------------------------------------------
# datasets that already carry an authorship boundary
# --------------------------------------------------------------------------
# SemEval-2024 Task 8 subtask C gives a real human prefix followed by a real
# machine continuation, with the change point as a word index. That is exactly
# the heterogeneous mixed case, with ground truth -- strictly better than the
# synthetic splices we build ourselves, which only approximate it.


def boundary_documents(
    n: int = 20000, min_words: int = 60, seed: int = 0, progress: bool = True
) -> list[dict]:
    """Yield Document-shaped dicts with exact human/AI character spans."""
    from datasets import load_dataset

    rng = random.Random(f"{seed}:semeval_c")
    reservoir = _Reservoir(n, rng)
    dataset = load_dataset("d0rj/SemEval2024-task8", "subtaskC", split="train", streaming=True)

    bar = tqdm(desc="hf:semeval_c", unit="row", disable=not progress)
    iterator = iter(dataset)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception:
            break
        bar.update(1)
        text = (row.get("text") or "").strip()
        try:
            cut = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        words = text.split()
        if not text or cut <= 0 or cut >= len(words) or len(words) < min_words:
            continue
        # Map the word index onto a character offset in the original string.
        offset, seen_words = 0, 0
        for match in re.finditer(r"\S+", text):
            seen_words += 1
            if seen_words == cut:
                offset = match.start()
                break
        if offset <= 0:
            continue
        reservoir.offer(
            {
                "id": f"semeval-c-{row.get('id')}",
                "text": text,
                "spans": [[0, offset, 0], [offset, len(text), 2]],
                "humanizer": 3,
                "source": "semeval/boundary",
                "meta": {"generator": "semeval-mixed", "boundary_word": cut},
            }
        )
    bar.close()
    return reservoir.items
