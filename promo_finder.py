#!/usr/bin/env python3
"""Find 20-60 second promo clips in timestamped microdrama screenplays.

Rules generate and validate candidates. An optional LLM reranks a shortlist;
its response can select candidate IDs but cannot alter their timestamps.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import glob
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

MIN_LEN, MAX_LEN = 20.0, 60.0
MAX_PROMOS_PER_EPISODE = 2
RAW_SCORE_MAX = 19.25  # prior max plus a 0.75 bonus for a strong physical opening

NON_SPEECH = {"CAMERA", "ACTION", "SOUND FX", "SFX", "SOUND", "MUSIC", "TRANSITION"}
STAKES = re.compile(
    r"marry|married|wedding|betray|abandon|dead|death|kill|pregnan|divorce|secret|"
    r"\blie[sd]?\b|truth|never|swore|regret|forgive|hate|love|worth|nothing|shame|"
    r"\bring\b|blood|poison|murder|traitor|expose|leave|dignity|mine|mistake", re.I
)
CHARGED = re.compile(
    r"cold|cruel|outrag|accus|furious|defian|scorn|shock|gasp|devastat|crush|betray|menac|"
    r"threat|mock|gloat|triumph|vicious|desperate|solemn|resolute|bitter|thunder|danger", re.I
)
IMPACT = re.compile(
    r"slap|strike|punch|shove|drag|seiz|grab|shatter|collaps|crash|"
    r"drops?\b|hurl|throw|thud|scream|blood|gun|knife|knock(?:s|ed)?\s+(?:him|her|them)?\s*down", re.I
)
CUE = re.compile(r"cliffhanger|ominous|swell|cut to black|fade|holds on|crescendo", re.I)
RESOLVED = re.compile(r"thank you|goodbye|farewell|forgive me|very well|agreed|i accept|\byes\b", re.I)
CLIFF_REVEAL = re.compile(r"accept my hand|marry me|marry you|coming home|return from the dead|i have proof|i know who|the same men .* ruling|i am offering it back", re.I)
STRONG_IMPACT = re.compile(r"\bpunch|\bslap|\bstrike|\bthud|knock(?:s|ed)?\s+(?:him|her|them)?\s*down|collaps", re.I)
DIRECT = re.compile(r"^(you|don't|do not|take|let|stop|get|look|who|how|what|why|where|tell|say)\b", re.I)

# A process-local cache: it stores model responses, not screenplay text.
# The SHA-256 key includes provider, model, endpoint and the complete prompt.
_RESPONSE_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def to_sec(s: str) -> float:
    parts = [float(p) for p in s.strip("() ").split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


def mmss(t: float) -> str:
    t = max(t, 0)
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


BLOCK = re.compile(r"^\*\*(.+?)\*\*\s*$")
TIME = re.compile(r"start:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*end:\s*\(?\s*([\d:.]+)\s*\)?", re.I)
SECTION = re.compile(r"^(SCENE\b|Scene (heading|start|end)|---|Actions and dialogue)", re.I)


def load_episodes(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return load_text(f.read(), path)


def load_text(text: str, path: str = "upload.txt") -> list[dict[str, Any]]:
    """One file may hold one episode or many (split on EPISODE NUMBER headers)."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.upper().startswith("EPISODE NUMBER")]
    if len(heads) < 2:
        return [parse_lines(lines, path)]
    heads.append(len(lines))
    return [parse_lines(lines[a:b], path) for a, b in zip(heads, heads[1:])]


def parse_episode(path: str) -> dict[str, Any]:
    return load_episodes(path)[0]


def parse_lines(lines: list[str], path: str) -> dict[str, Any]:
    ep: dict[str, Any] = {"path": path, "number": None, "logline": "", "continuity": "", "events": []}
    for ln in lines[:15]:
        up = ln.upper()
        if up.startswith("EPISODE NUMBER"):
            m = re.search(r"\d+", ln)
            ep["number"] = int(m.group()) if m else None
        elif up.startswith("EPISODE LOGLINE"):
            ep["logline"] = ln.split(":", 1)[1].strip() if ":" in ln else ""
        elif up.startswith("CONTINUITY"):
            ep["continuity"] = ln.split(":", 1)[1].strip() if ":" in ln else ""

    if ep["number"] is None:
        m = re.search(r"(?:episode|ep)[-_ ]*(\d+)", os.path.basename(path), re.I) or re.search(r"(\d+)", os.path.basename(path))
        ep["number"] = int(m.group(1)) if m else 0

    i = 0
    while i < len(lines):
        m = BLOCK.match(lines[i].strip())
        t = TIME.search(lines[i + 1]) if m and i + 1 < len(lines) else None
        if not t:
            i += 1
            continue
        label = m.group(1).strip().rstrip(":").strip()
        tone = re.findall(r"\(([^()]*)\)", label)
        kind = re.sub(r"\([^()]*\)", "", label).strip().rstrip(":").strip().upper()
        content, j = [], i + 2
        while j < len(lines) and not BLOCK.match(lines[j].strip()) and not SECTION.match(lines[j].strip()):
            if lines[j].strip():
                content.append(lines[j].strip())
            elif content:
                break
            j += 1
        up = label.upper()
        ep["events"].append({
            "start": to_sec(t.group(1)),
            "end": to_sec(t.group(2)),
            "kind": kind,
            "speech": kind not in NON_SPEECH,
            "tone": tone[-1] if tone and not re.fullmatch(r"O\.?S\.?|V\.?O\.?|OFF-SCREEN", tone[-1], re.I) else "",
            "offscreen": any(w in up for w in ("O.S", "OFF-SCREEN", "V.O", "VOICE")),
            "text": " ".join(content).strip('"“” '),
        })
        i = j
    ep["events"].sort(key=lambda e: (e["start"], e["end"]))
    ep["length"] = max((e["end"] for e in ep["events"]), default=0.0)
    return ep


def within(ep: dict[str, Any], a: float, b: float) -> list[dict[str, Any]]:
    return [e for e in ep["events"] if e["start"] >= a - 0.05 and e["end"] <= b + 0.05]


def hook_score(evs: list[dict[str, Any]], window_start: float) -> tuple[float, str, str]:
    """Score what a viewer encounters in the first three seconds, including action."""
    early = [e for e in evs if e["start"] < window_start + 3.0 and e["end"] > window_start]
    first_speech = next((e for e in early if e["speech"] and e["text"]), None)
    impact_events = [e for e in early if e["kind"] in {"ACTION", "SOUND FX", "SFX", "SOUND"} and IMPACT.search(e["text"])]
    score, why = 0.0, []

    if first_speech:
        txt = first_speech["text"]
        if "?" in txt or "!" in txt:
            score += 1.0; why.append("punctuation")
        if len(txt.split()) <= 9:
            score += 1.0; why.append("short spoken hook")
        if STAKES.search(txt):
            score += 1.5; why.append("high-stakes words")
        if CHARGED.search(first_speech["tone"]):
            score += 1.0; why.append("charged delivery")
        if DIRECT.match(txt):
            score += 0.5; why.append("direct address")
        if txt.startswith("...") or txt[:1].islower():
            score -= 2.0; why.append("starts mid-sentence")
        if first_speech["offscreen"] or first_speech["kind"].startswith(("OFF", "VOICE", "UNKNOWN")):
            score -= 1.0; why.append("unseen speaker")
        hook = f'{first_speech["kind"].title()}: "{txt}"'
    else:
        hook = ""

    if impact_events:
        score += 2.5
        why.append("physical action in first 3 seconds")
        action = impact_events[0]["text"].strip()
        action_hook = f"Action: {action[:180]}"
        if first_speech:
            hook = f"{action_hook} | First line: {first_speech['kind'].title()}: \"{first_speech['text']}\""
        else:
            hook = action_hook

    if not first_speech and not impact_events:
        return 0.0, "", "no clear spoken or physical hook in first 3 seconds"

    return max(min(score, 5.0), 0.0), hook, ", ".join(why)


def cliff_score(evs: list[dict[str, Any]], ep_len: float) -> tuple[float, str, str]:
    line = next((e for e in reversed(evs) if e["speech"] and e["text"]), None)
    last = evs[-1] if evs else None
    score, why = 0.0, []
    if last and abs(last["end"] - ep_len) < 0.6:
        score += 2.5; why.append("episode ends here")
    if any(CUE.search(e["text"]) for e in evs[-3:] if not e["speech"]):
        score += 1.5; why.append("score/fade cue")
    if line:
        txt = line["text"].rstrip()
        if txt.endswith("?"):
            score += 2.0; why.append("ends on a question")
        if txt.endswith("..."):
            score += 1.5; why.append("trails off")
        if STAKES.search(txt):
            score += 1.0; why.append("high-stakes last line")
        if CLIFF_REVEAL.search(txt):
            score += 3.5; why.append("major reveal or proposal")
        if RESOLVED.search(txt):
            score -= 1.5; why.append("sounds resolved")
        if last and line["end"] < last["end"] - 4:
            score -= 1.0
    hook_text = f'{line["kind"].title()}: "{line["text"]}"' if line else (last["text"][:120] if last else "")
    return max(min(score, 5.0), 0.0), hook_text, ", ".join(why)


def intensity(evs: list[dict[str, Any]], dur: float) -> float:
    speech = [e for e in evs if e["speech"]]
    if not speech:
        return 0.0
    charged = sum(bool(CHARGED.search(e["tone"])) for e in speech) / len(speech)
    pace = min(len(speech) / max(dur, 1) / 0.35, 1.0)
    voices = len({e["kind"] for e in speech if not e["offscreen"]})
    return min(2.5 * charged + 2.0 * pace + (0.5 if voices >= 2 else -0.5), 4.0)


def is_physical_start(e: dict[str, Any]) -> bool:
    return (not e["speech"]) and e["kind"] in {"ACTION", "SOUND FX", "SFX", "SOUND"} and bool(IMPACT.search(e["text"]))


def strong_opening_impact_bonus(evs: list[dict[str, Any]], window_start: float) -> float:
    """Give a modest extra reward to an actual hit/collapse, not generic movement."""
    return 0.75 if any(
        e["kind"] in {"ACTION", "SOUND FX", "SFX", "SOUND"}
        and e["start"] < window_start + 3.0 and e["end"] > window_start
        and STRONG_IMPACT.search(e["text"])
        for e in evs
    ) else 0.0


def windows(ep: dict[str, Any], lo: float, hi: float):
    """Generate windows starting on dialogue or a described physical-impact beat."""
    starts = [e for e in ep["events"] if e["speech"] or is_physical_start(e)]
    # If a sound-effect cue describes the same impact as an ACTION block, start
    # on the visual action timestamp rather than trimming its first half-second.
    visual_impacts = [e for e in starts if e["kind"] == "ACTION" and is_physical_start(e)]
    starts = [e for e in starts if not (
        e["kind"] in {"SOUND FX", "SFX", "SOUND"}
        and any(abs(e["start"] - action["start"]) <= 0.75 for action in visual_impacts)
    )]
    ends = [e for e in ep["events"] if e["speech"] or e["kind"] in ("ACTION", "SOUND FX", "SFX")]
    seen = set()
    for start in starts:
        for end in ends:
            duration = end["end"] - start["start"]
            if lo <= duration <= hi and end["end"] > start["start"]:
                key = (round(start["start"], 2), round(end["end"], 2))
                if key not in seen:
                    seen.add(key)
                    yield key


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(inter, 0.0) / max(min(a[1] - a[0], b[1] - b[0]), 0.01)


def temporal_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0.0


def too_similar(candidate: dict[str, Any], kept: dict[str, Any], overlap_limit: float, min_gap_seconds: float) -> bool:
    if overlap(candidate["win"], kept["win"]) >= overlap_limit:
        return True
    return temporal_gap(candidate["win"], kept["win"]) < min_gap_seconds


def thin(items: list[dict[str, Any]], key, n: int, limit: float = 0.3, min_gap_seconds: float = 0.0) -> list[dict[str, Any]]:
    """Greedy diversity filter, optionally requiring temporal separation."""
    kept = []
    for item in sorted(items, key=key, reverse=True):
        if all(not too_similar(item, old, limit, min_gap_seconds) for old in kept):
            kept.append(item)
        if len(kept) == n:
            break
    return kept


def build(ep: dict[str, Any], early_bonus: dict[int, float], stitch_cost: float) -> list[dict[str, Any]]:
    out = []
    for a, b in windows(ep, MIN_LEN, MAX_LEN):
        evs = within(ep, a, b)
        hook, hook_text, hook_why = hook_score(evs, a)
        cliff, cliff_text, cliff_why = cliff_score(evs, ep["length"])
        duration = b - a
        visual_bonus = strong_opening_impact_bonus(evs, a)
        raw_score = (1.3 * hook + 1.4 * cliff + intensity(evs, duration)
                     + early_bonus.get(ep["number"], 0.0) + (0.5 if 30 <= duration <= 50 else 0.0)
                     + visual_bonus)
        out.append({
            "win": (a, b), "segs": [(a, b)], "h": hook, "k": cliff,
            "i": intensity(evs, duration), "hook": hook_text, "hook_why": hook_why,
            "cliff": cliff_text, "cliff_why": cliff_why, "stitched": False,
            "rule_score_raw": round(raw_score, 3), "visual_bonus": visual_bonus,
        })

    # A two-part cut is considered only if the opener and ending cannot fit in one 60s cut.
    hooks, cliffs = [], []
    for a, b in windows(ep, 8, 26):
        evs = within(ep, a, b)
        h = hook_score(evs, a)
        k = cliff_score(evs, ep["length"])
        hooks.append({"win": (a, b), "v": h})
        cliffs.append({"win": (a, b), "v": k})
    hooks = thin(hooks, lambda x: x["v"][0], 6, limit=0.3, min_gap_seconds=3.0)
    cliffs = thin(cliffs, lambda x: x["v"][0], 6, limit=0.3, min_gap_seconds=3.0)
    for hook_candidate in hooks:
        for cliff_candidate in cliffs:
            (a, b), (c, d) = hook_candidate["win"], cliff_candidate["win"]
            total = (b - a) + (d - c)
            if c >= b + 1 and d - a > MAX_LEN and MIN_LEN <= total <= MAX_LEN:
                evs = within(ep, a, b) + within(ep, c, d)
                hook_data, cliff_data = hook_candidate["v"], cliff_candidate["v"]
                visual_bonus = strong_opening_impact_bonus(within(ep, a, b), a)
                raw_score = (1.3 * hook_data[0] + 1.4 * cliff_data[0] + intensity(evs, total)
                             + early_bonus.get(ep["number"], 0.0)
                             + (0.5 if 30 <= total <= 50 else 0.0) + visual_bonus - stitch_cost)
                out.append({
                    "win": (a, d), "segs": [(a, b), (c, d)], "h": hook_data[0], "k": cliff_data[0],
                    "i": intensity(evs, total), "hook": hook_data[1], "hook_why": hook_data[2],
                    "cliff": cliff_data[1], "cliff_why": cliff_data[2], "stitched": True,
                    "rule_score_raw": round(raw_score, 3), "visual_bonus": visual_bonus,
                })

    for candidate in out:
        candidate["episode"] = ep["number"]
        candidate["duration"] = sum(b - a for a, b in candidate["segs"])
        candidate["prior"] = early_bonus.get(ep["number"], 0.0)
        candidate["score"] = round(max(0.0, min(10.0, candidate["rule_score_raw"] / RAW_SCORE_MAX * 10.0)), 3)
    return out


# ---------- Optional, swappable LLM ----------
PROMPT = """You select short promotional clips for vertical microdrama ads.
A good promo must: (1) have an attention-grabbing hook within its first 3 seconds,
(2) end on an unresolved cliffhanger, and (3) make sense to a stranger who has
never seen the show. Judge the complete candidate transcript, not just its first
and last lines. Action and sound descriptions are part of the transcript.

Episode {num}
Logline: {logline}
Continuity notes (context, not assumed viewer knowledge): {continuity}

Candidates (IDs are fixed; '+' joins separate segments in playback order):
{cands}

Return ONLY a JSON array. Include only candidates worth posting. Every returned
object must have: id (integer candidate ID), score (number 1-10), hook (brief
transcript-grounded description of what grabs attention in the first 3 seconds),
cliffhanger (brief description grounded in the final moment), why (one sentence),
and standalone_ok (boolean). Set standalone_ok=false when a first-time viewer
would need earlier episode knowledge. Do not invent dialogue, characters, events,
or timestamps. Do not modify candidate IDs. If none qualify, return []."""


def _model_settings(provider: str, config: dict[str, Any] | None) -> tuple[str, str, str]:
    config = config or {}
    model = str(config.get("model") or os.environ.get("LLM_MODEL") or "").strip()
    base = str(config.get("base_url") or os.environ.get("LLM_BASE_URL") or ("https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1")).rstrip("/")
    key = str(config.get("api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY") or "")
    return model, base, key


def ask_model(prompt: str, provider: str, config: dict[str, Any] | None = None) -> str:
    model, base, key = _model_settings(provider, config)
    if not key:
        raise ValueError("No API key supplied")
    if not model:
        raise ValueError("No model ID supplied")
    if provider == "anthropic":
        url = base + ("/v1/messages" if not base.endswith("/v1") else "/messages")
        body = {"model": model, "max_tokens": 3000, "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        pick = lambda response: response["content"][0]["text"]
    elif provider == "openai":
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        body = {"model": model, "temperature": 0.1, "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": "Bearer " + key, "content-type": "application/json"}
        pick = lambda response: response["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    request = urllib.request.Request(url, json.dumps(body).encode("utf-8"), headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return pick(payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc


def _extract_json_array(raw: str) -> list[Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    if start < 0:
        raise ValueError("No JSON array found")
    depth, quoted, escaped = 0, False, False
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("Unterminated JSON array")
    result = json.loads(text[start:end])
    if not isinstance(result, list):
        raise ValueError("Model response must be a JSON array")
    return result


def transcript(ep: dict[str, Any], candidate: dict[str, Any]) -> str:
    rows = []
    for a, b in candidate["segs"]:
        for event in within(ep, a, b):
            if event["speech"]:
                rows.append(f'[{mmss(event["start"])}] {event["kind"].title()}: {event["text"]}')
            elif event["kind"] in ("ACTION", "SOUND FX", "SFX"):
                rows.append(f'[{mmss(event["start"])}] ({event["kind"]}: {event["text"][:180]})')
    return "\n".join(rows)


def rerank(ep: dict[str, Any], shortlist: list[dict[str, Any]], provider: str,
           config: dict[str, Any] | None = None, ask=ask_model, retries: int = 2
           ) -> tuple[list[dict[str, Any]], str, str]:
    cands = "\n\n".join(
        f'#{i} ' + " + ".join(f"{mmss(a)}-{mmss(b)}" for a, b in c["segs"]) + f'\n{transcript(ep, c)}'
        for i, c in enumerate(shortlist)
    )
    prompt = PROMPT.format(num=ep["number"], logline=ep["logline"], continuity=ep["continuity"], cands=cands)
    model, base, _ = _model_settings(provider, config)
    cache_key = hashlib.sha256((provider + "\0" + model + "\0" + base + "\0" + prompt).encode("utf-8")).hexdigest()
    with _CACHE_LOCK:
        raw = _RESPONSE_CACHE.get(cache_key)
    cache_hit = raw is not None

    if raw is None:
        last_error = "Unknown model error"
        for attempt in range(retries):
            try:
                raw = ask(prompt, provider, config)
                break
            except Exception as err:  # do not fail the full batch because one model call failed
                last_error = str(err)
                if attempt + 1 < retries:
                    time.sleep(0.5 * (attempt + 1))
        else:
            return shortlist, "fallback", last_error

    try:
        data = _extract_json_array(raw)
        accepted: list[dict[str, Any]] = []
        seen_ids = set()
        invalid_rows = 0
        for row in data:
            if not isinstance(row, dict):
                invalid_rows += 1
                continue
            try:
                idx = int(row["id"])
                llm_score = float(row["score"])
                standalone = row["standalone_ok"]
                if isinstance(row["id"], bool) or not 0 <= idx < len(shortlist):
                    invalid_rows += 1
                    continue
                if not isinstance(standalone, bool) or not (1.0 <= llm_score <= 10.0):
                    invalid_rows += 1
                    continue
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
                continue
            if idx in seen_ids:
                continue
            seen_ids.add(idx)
            if not standalone:
                continue
            candidate = dict(shortlist[idx])
            # Both rule-only and LLM-assisted final scores use the same 0-10 scale.
            # A 35% rule prior stabilizes ranking; the LLM contributes 65% semantic judgment.
            llm_normalized = llm_score
            candidate["llm_score"] = round(llm_score, 2)
            candidate["score"] = round(0.35 * candidate["score"] + 0.65 * llm_normalized, 3)
            candidate["hook"] = str(row.get("hook") or candidate["hook"])[:500]
            candidate["cliff"] = str(row.get("cliffhanger") or candidate["cliff"])[:500]
            candidate["why"] = str(row.get("why") or "LLM accepted the candidate after semantic review")[:700]
            candidate["scorer"] = provider
            accepted.append(candidate)
        if invalid_rows and not accepted and data:
            return shortlist, "fallback", "Model response did not contain valid candidate records"
        # Cache only syntactically and semantically valid responses. A transient
        # malformed answer should not poison later reruns in this process.
        if not cache_hit:
            with _CACHE_LOCK:
                _RESPONSE_CACHE[cache_key] = raw
        if not accepted:
            # A valid empty list or all candidates explicitly rejected is a real rejection.
            return [], "all_rejected", "The model rejected all shortlisted candidates as unsuitable or not standalone"
        return accepted, ("cache" if cache_hit else "ok"), ""
    except Exception as err:
        return shortlist, "fallback", f"Invalid model response: {err}"


def valid(candidate: dict[str, Any], ep_len: float) -> bool:
    previous_end = -1.0
    for a, b in candidate["segs"]:
        if not (0 <= a < b <= ep_len + 0.5 and a >= previous_end):
            return False
        previous_end = b
    duration = sum(b - a for a, b in candidate["segs"])
    return MIN_LEN - 0.5 <= duration <= MAX_LEN + 0.5


def explain(candidate: dict[str, Any]) -> str:
    parts = []
    if candidate["stitched"]:
        parts.append("two-part cut: opener and ending are too far apart for one continuous 60-second clip")
    if candidate["hook_why"]:
        parts.append("hook: " + candidate["hook_why"])
    if candidate["cliff_why"]:
        parts.append("ending: " + candidate["cliff_why"])
    if candidate["prior"]:
        parts.append("small early-episode prior")
    if candidate.get("visual_bonus"):
        parts.append("strong physical impact in the opening")
    return "; ".join(parts) or "scored on pacing and conflict"


def _process_episode(item: tuple[dict[str, Any], list[dict[str, Any]]], llm: str,
                     llm_config: dict[str, Any] | None) -> tuple[int, list[dict[str, Any]], str, str]:
    ep, shortlist = item
    if llm == "none":
        return ep["number"], shortlist, "rules", ""
    reranked, status, message = rerank(ep, shortlist, llm, llm_config)
    if status == "fallback":
        for candidate in reranked:
            candidate["scorer"] = "rules (LLM fallback)"
    return ep["number"], reranked, status, message


def run_pipeline(eps: list[dict[str, Any]], top_per_ep: int = 2, early_bonus: float = 0.5,
                 stitch_cost: float = 1.5, llm: str = "none", min_score: float = 0.0,
                 verbose: bool = False, llm_config: dict[str, Any] | None = None,
                 llm_workers: int = 4, return_meta: bool = False):
    if not 1 <= top_per_ep <= MAX_PROMOS_PER_EPISODE:
        raise ValueError(f"top_per_ep must be between 1 and {MAX_PROMOS_PER_EPISODE}")
    if not 0.0 <= min_score <= 10.0:
        raise ValueError("min_score must be between 0 and 10")
    if llm not in {"none", "openai", "anthropic"}:
        raise ValueError("llm must be none, openai or anthropic")

    eps = sorted(eps, key=lambda e: e["number"])
    early = {e["number"]: early_bonus for e in eps if 1 <= e["number"] <= 3}
    prepared = []
    candidate_counts = {}
    for ep in eps:
        candidates = build(ep, early, stitch_cost)
        candidate_counts[ep["number"]] = len(candidates)
        shortlist = thin(candidates, lambda c: c["score"], 8, limit=0.35, min_gap_seconds=0.0)
        prepared.append((ep, shortlist))

    processed: dict[int, tuple[list[dict[str, Any]], str, str]] = {}
    if llm == "none" or len(prepared) <= 1:
        for item in prepared:
            ep_num, selected, status, message = _process_episode(item, llm, llm_config)
            processed[ep_num] = (selected, status, message)
    else:
        workers = max(1, min(int(llm_workers), len(prepared), 8))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_episode, item, llm, llm_config) for item in prepared]
            for future in concurrent.futures.as_completed(futures):
                ep_num, selected, status, message = future.result()
                processed[ep_num] = (selected, status, message)

    promos = []
    statuses: dict[str, int] = {}
    warnings = []
    ep_by_num = {ep["number"]: ep for ep in eps}
    for ep_num in sorted(ep_by_num):
        ep = ep_by_num[ep_num]
        selected, status, message = processed[ep_num]
        statuses[status] = statuses.get(status, 0) + 1
        if message:
            warnings.append({"episode": ep_num, "status": status, "message": message})
        picks = [c for c in thin(selected, lambda c: c["score"], top_per_ep,
                                 limit=0.35, min_gap_seconds=10.0)
                 if valid(c, ep["length"]) and c["score"] >= min_score]
        for candidate in picks:
            candidate.setdefault("scorer", "rules")
            candidate["why"] = candidate.get("why") or explain(candidate)
        promos.extend(picks)
        if verbose:
            print(f"ep {ep_num}: {candidate_counts[ep_num]} candidates -> {len(picks)} promos ({status})")

    promos.sort(key=lambda candidate: -candidate["score"])
    meta = {
        "episodes": len(eps),
        "candidate_counts": candidate_counts,
        "llm_status_counts": statuses,
        "warnings": warnings,
        "cache_entries": len(_RESPONSE_CACHE),
    }
    return (promos, meta) if return_meta else promos


def write_outputs(promos: list[dict[str, Any]], outdir: str, n_eps: int,
                  tier_a: int = 10, tier_b: int = 30) -> None:
    os.makedirs(outdir, exist_ok=True)
    output_rows = []
    for rank, promo in enumerate(promos, 1):
        promo["rank"] = rank
        promo["tier"] = "A (review first)" if rank <= tier_a else "B (backup)" if rank <= tier_b else "C (long tail)"
        promo["why"] = promo.get("why") or explain(promo)
        promo["segments"] = [{"episode": promo["episode"], "start": mmss(a), "end": mmss(b)} for a, b in promo["segs"]]
        output_rows.append({
            "rank": promo["rank"], "tier": promo["tier"], "episode": promo["episode"],
            "segments": promo["segments"], "duration": round(promo["duration"], 2),
            "score": promo["score"], "rule_score_raw": promo["rule_score_raw"],
            "llm_score": promo.get("llm_score"), "hook": promo["hook"], "cliff": promo["cliff"],
            "why": promo["why"], "stitched": promo["stitched"], "scorer": promo.get("scorer", "rules"),
            "parts": {"hook": promo["h"], "cliffhanger": promo["k"], "intensity": round(promo["i"], 2),
                      "early_prior": promo["prior"], "strong_opening_impact_bonus": promo.get("visual_bonus", 0.0)},
        })
    with open(os.path.join(outdir, "promos.json"), "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "promos.md"), "w", encoding="utf-8") as f:
        f.write(f"# Promos ({len(promos)} from {n_eps} episodes)\n\n")
        f.write("**Post first (top 3):**\n")
        for promo in promos[:3]:
            segments = " + ".join(f'{s["start"]}-{s["end"]}' for s in promo["segments"])
            f.write(f'- #{promo["rank"]} Ep {promo["episode"]} {segments}: {promo["why"]}\n')
        f.write(f"\nTier A = ranks 1-{tier_a}, B = ranks {tier_a + 1}-{tier_b}, C = the rest. "
                "Rule-based scores are a proxy; check context and visuals before posting.\n\n")
        current_tier = None
        for promo in promos:
            if promo["tier"] != current_tier:
                current_tier = promo["tier"]
                f.write(f"# {current_tier}\n\n")
            segments = " + ".join(f'{s["start"]}-{s["end"]}' for s in promo["segments"])
            f.write(f'## #{promo["rank"]} Ep {promo["episode"]} {segments} '
                    f'({promo["duration"]:.1f}s, score {promo["score"]:.2f}/10)\n'
                    f'- Hook: {promo["hook"]}\n- Cliffhanger: {promo["cliff"]}\n- Why: {promo["why"]}\n\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Find promo clips in timestamped screenplay files")
    parser.add_argument("indir", help="Folder containing .txt/.md files, or one merged file")
    parser.add_argument("--out", default="out")
    parser.add_argument("--llm", default="none", choices=["none", "anthropic", "openai"])
    parser.add_argument("--top-per-ep", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-score", type=float, default=0.0, help="drop weak promos; normalized 0-10 score")
    parser.add_argument("--early-bonus", type=float, default=0.5, help="small prior for episodes 1-3 (0 disables)")
    parser.add_argument("--tier-a", type=int, default=10)
    parser.add_argument("--stitch-cost", type=float, default=1.5)
    parser.add_argument("--llm-workers", type=int, default=4, help="maximum concurrent episode-level LLM calls")
    args = parser.parse_args()

    paths = sorted(p for ext in ("*.txt", "*.md") for p in glob.glob(os.path.join(args.indir, ext))) if os.path.isdir(args.indir) else [args.indir]
    episodes = []
    for path in paths:
        try:
            found = load_episodes(path)
        except Exception as err:
            print(f"skip {path}: {err}", file=sys.stderr)
            continue
        for ep in found:
            if ep["events"]:
                episodes.append(ep)
            else:
                print(f"skip {os.path.basename(path)} ep {ep['number']}: no timestamped events", file=sys.stderr)
    episodes.sort(key=lambda e: e["number"])
    promos, meta = run_pipeline(episodes, args.top_per_ep, args.early_bonus, args.stitch_cost,
                                args.llm, args.min_score, verbose=True, llm_workers=args.llm_workers,
                                return_meta=True)
    write_outputs(promos, args.out, len(episodes), args.tier_a, args.tier_a * 3)
    print(f"Parsed {meta['episodes']} episodes; generated {len(promos)} promos in {args.out}/")
    if meta["warnings"]:
        print(f"LLM warnings/fallbacks: {len(meta['warnings'])}", file=sys.stderr)
        for warning in meta["warnings"]:
            print(f"  Episode {warning['episode']}: {warning['status']} - {warning['message']}", file=sys.stderr)


if __name__ == "__main__":
    main()
