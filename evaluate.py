#!/usr/bin/env python3
"""Evaluate ranked promos against a small human-labelled gold CSV.

Gold CSV columns: episode,start,end. Example times: 01:14.40 and 01:50.00.
A prediction is a hit when temporal IoU with the gold clip is >= --iou.
For a two-part prediction, segment durations are evaluated as a union.
"""
import argparse
import csv
import json
from pathlib import Path


def sec(value: str) -> float:
    parts = [float(x) for x in value.strip().split(":")]
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return total


def intersection(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou_segments(segments, gold):
    overlap = sum(intersection(segment, gold) for segment in segments)
    predicted = sum(end - start for start, end in segments)
    union = predicted + (gold[1] - gold[0]) - overlap
    return overlap / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="promos.json")
    parser.add_argument("gold_csv", help="CSV with episode,start,end")
    parser.add_argument("--iou", type=float, default=0.5, help="minimum temporal IoU for a hit")
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    by_episode = {}
    for row in predictions:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    gold = []
    with open(args.gold_csv, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            gold.append((int(row["episode"]), (sec(row["start"]), sec(row["end"]))))

    if not gold:
        raise SystemExit("Gold CSV contains no labelled episodes")

    top1_hits, top2_hits, best_ious = 0, 0, []
    for episode, gold_window in gold:
        ranked = sorted(by_episode.get(episode, []), key=lambda row: int(row["rank"]))
        scores = {}
        for prediction in ranked:
            segments = [(sec(seg["start"]), sec(seg["end"])) for seg in prediction["segments"]]
            scores[int(prediction["rank"])] = iou_segments(segments, gold_window)
        best = max(scores.values(), default=0.0)
        top1 = scores.get(1, 0.0) >= args.iou
        top2 = any(scores.get(rank, 0.0) >= args.iou for rank in (1, 2))
        top1_hits += int(bool(top1))
        top2_hits += int(bool(top2))
        best_ious.append(best)
        print(f"Episode {episode}: best IoU={best:.3f}; top-1={'HIT' if top1 else 'MISS'}; top-2={'HIT' if top2 else 'MISS'}")

    n = len(gold)
    print(f"\nLabelled episodes: {n}")
    print(f"Top-1 hit rate (rank 1): {top1_hits}/{n} = {top1_hits/n:.1%}")
    print(f"Top-2 hit rate (ranks 1-2): {top2_hits}/{n} = {top2_hits/n:.1%}")
    print(f"Mean best temporal IoU: {sum(best_ious)/n:.3f}")


if __name__ == "__main__":
    main()
