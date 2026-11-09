# Promo Finder

A small tool for finding 20-60 second promotional moments in timestamped microdrama screenplays. The deterministic pipeline parses the screenplay, generates candidate clips, scores their opening hook / ending / conflict, and exports ranked JSON and Markdown. An optional LLM reviews the top eight candidates per episode for semantic quality and standalone comprehension.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload one screenplay per file or one merged file with `EPISODE NUMBER:` headers. The included `sample_input/demo_episode_1.txt` is synthetic and safe to use as a smoke test.

CLI alternative:

```bash
python promo_finder.py sample_input/ --out out
# Or pass one merged screenplay file:
python promo_finder.py /path/to/merged_screenplays.txt --out out
```

## Optional LLM reranking

Choose OpenAI-compatible or Anthropic in the Streamlit sidebar. Enter the provider-supported model ID, API key and API base URL. For CLI use, set `LLM_API_KEY`, `LLM_MODEL`, and optionally `LLM_BASE_URL`, then run:

```bash
python promo_finder.py /path/to/screenplays.txt --out out --llm openai --llm-workers 4
```

The model sees the logline, continuity notes and transcripts for the top eight rule-based candidates per episode. It returns candidate IDs and semantic judgments, not timestamps. The pipeline retains the original timestamps and validates every selected segment. API errors or malformed responses fall back to rules; a valid response rejecting every candidate results in no promo for that episode. Calls run concurrently (default four workers, maximum eight) and exact provider/model/prompt responses are cached in process memory using a SHA-256 key. The cache is not persistent across process restarts.

## Scoring and safeguards

Rule-only and LLM-assisted final scores both use a 0-10 scale. LLM mode combines 35% normalized rule score with 65% model score. Rule scoring includes the first-three-second spoken or physical hook, cliffhanger cues, conflict intensity, a small duration preference, a small early-episode prior, a modest strong-impact bonus and a penalty for two-part cuts. The pipeline returns at most two promos per episode and applies overlap / temporal-separation filtering. `--min-score` accepts a normalized 0-10 threshold.

## Tests and evaluation

```bash
python -m unittest -v
```

The tests cover parsing, timestamp variants, action hooks, LLM rejection / fallback behavior, normalized scores, response caching and output constraints. To run the same tests against a full merged input pack, set `PROMO_TEST_INPUT` to the path of that file:

```bash
PROMO_TEST_INPUT=/path/to/merged_screenplays.txt python -m unittest -v
```

To measure agreement with a hand-labelled sample, copy `gold_template.csv`, add one row per labelled episode (`episode,start,end`), then run:

```bash
python evaluate.py out/promos.json gold.csv --iou 0.5
```

The evaluator reports top-1 and top-2 hit rates using temporal intersection-over-union (IoU). It does not create ground truth; labels must be chosen manually before examining the model's picks to reduce confirmation bias.

## Important limitations

- The tool sees screenplay text and action descriptions, not the actual video, actor performance, shot quality, sound mix or edit pacing.
- English keyword heuristics are proxies, not a measure of audience engagement.
- An LLM standalone check is optional and still needs editor review; no provider call is made in rules-only mode.
- A valid 20-60 second segment is not automatically a good promo. Review the clip against the video before posting.
- The repository contains only synthetic sample data. Do not add Kuku's confidential screenplays or dialogue-containing raw outputs to GitHub.
