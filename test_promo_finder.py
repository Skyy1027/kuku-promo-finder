import json
import unittest
from unittest.mock import Mock

import promo_finder as pf


class PromoFinderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        fixture = os.environ.get("PROMO_TEST_INPUT", "sample_input/demo_episode_1.txt")
        cls.episodes = pf.load_episodes(fixture)
        cls.full_pack_test = len(cls.episodes) == 50

    def setUp(self):
        pf._RESPONSE_CACHE.clear()

    def test_merged_pack_parses(self):
        self.assertGreaterEqual(len(self.episodes), 1)
        self.assertGreater(sum(len(ep["events"]) for ep in self.episodes), 0)
        if self.full_pack_test:
            self.assertEqual(len(self.episodes), 50)
            self.assertEqual(sum(len(ep["events"]) for ep in self.episodes), 3727)

    def test_time_formats(self):
        self.assertAlmostEqual(pf.to_sec("01:02.30"), 62.30)
        self.assertAlmostEqual(pf.to_sec("00:01:02.30"), 62.30)

    def test_alternate_timestamp_format_and_multiline_dialogue(self):
        lines = [
            "SCENE 01",
            "**NARRATOR (Urgent):**",
            "Start: 00:00:00.00 | End: 00:00:02.00",
            "The first line continues",
            "on a second line.",
            "",
            "**ACTION:**",
            "Start: 00:00:02.00 | End: 00:00:04.50",
            "A door slams shut.",
        ]
        parsed = pf.parse_lines(lines, "episode-7.txt")
        self.assertEqual(parsed["number"], 7)
        self.assertEqual(len(parsed["events"]), 2)
        self.assertEqual(parsed["events"][0]["text"], "The first line continues on a second line.")
        self.assertAlmostEqual(parsed["events"][1]["end"], 4.5)

    def test_action_hook_is_a_candidate(self):
        events = [
            {"start": 0.0, "end": 1.5, "kind": "ACTION", "speech": False, "tone": "", "offscreen": False,
             "text": "The stranger punches the guard, who collapses to the floor."},
            {"start": 1.5, "end": 3.0, "kind": "SOUND FX", "speech": False, "tone": "", "offscreen": False,
             "text": "A heavy punch and thud."},
            {"start": 3.0, "end": 4.0, "kind": "CHARACTER", "speech": True, "tone": "Furious", "offscreen": False,
             "text": "What are you waiting for?!"},
            {"start": 25.0, "end": 26.0, "kind": "CHARACTER", "speech": True, "tone": "Resolute", "offscreen": False,
             "text": "I am offering it back."},
        ]
        ep = {"number": 99, "events": events, "length": 26.0, "logline": "", "continuity": ""}
        candidates = pf.build(ep, {}, 1.5)
        knockout = [c for c in candidates if not c["stitched"] and c["segs"][0][0] == 0.0 and c["segs"][0][1] == 26.0]
        self.assertTrue(knockout)
        self.assertIn("punch", knockout[0]["hook"].lower())
        self.assertGreater(knockout[0]["visual_bonus"], 0)

    def test_all_standalone_rejections_are_respected(self):
        ep = next(ep for ep in self.episodes if ep["number"] == 1)
        shortlist = pf.thin(pf.build(ep, {}, 1.5), lambda c: c["score"], 2)
        answer = json.dumps([{"id": i, "score": 8, "hook": "hook", "cliffhanger": "ending", "why": "reason", "standalone_ok": False}
                            for i in range(len(shortlist))])
        selected, status, _ = pf.rerank(ep, shortlist, "openai", {"model": "test"}, ask=lambda *_: answer)
        self.assertEqual(selected, [])
        self.assertEqual(status, "all_rejected")

    def test_api_error_falls_back_to_rules(self):
        ep = next(ep for ep in self.episodes if ep["number"] == 1)
        shortlist = pf.thin(pf.build(ep, {}, 1.5), lambda c: c["score"], 2)
        selected, status, message = pf.rerank(ep, shortlist, "openai", {"model": "test"},
                                                ask=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")), retries=1)
        self.assertEqual(status, "fallback")
        self.assertEqual(len(selected), len(shortlist))
        self.assertIn("offline", message)

    def test_malformed_response_falls_back(self):
        ep = next(ep for ep in self.episodes if ep["number"] == 1)
        shortlist = pf.thin(pf.build(ep, {}, 1.5), lambda c: c["score"], 2)
        selected, status, _ = pf.rerank(ep, shortlist, "openai", {"model": "test"}, ask=lambda *_: "not json", retries=1)
        self.assertEqual(status, "fallback")
        self.assertEqual(len(selected), len(shortlist))

    def test_llm_and_rules_share_normalized_score_scale(self):
        ep = next(ep for ep in self.episodes if ep["number"] == 1)
        shortlist = pf.thin(pf.build(ep, {}, 1.5), lambda c: c["score"], 1)
        original = shortlist[0]["score"]
        response = json.dumps([{"id": 0, "score": 4, "hook": "grounded hook", "cliffhanger": "grounded ending",
                                "why": "good story", "standalone_ok": True}])
        selected, status, _ = pf.rerank(ep, shortlist, "openai", {"model": "test"}, ask=lambda *_: response)
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(selected[0]["score"], round(0.35 * original + 0.65 * 4, 3))
        self.assertEqual(selected[0]["segs"], shortlist[0]["segs"])
        self.assertGreaterEqual(selected[0]["score"], 0)
        self.assertLessEqual(selected[0]["score"], 10)

    def test_malformed_response_is_not_cached(self):
        ep = next(ep for ep in self.episodes if ep["number"] == 1)
        shortlist = pf.thin(pf.build(ep, {}, 1.5), lambda c: c["score"], 1)
        _, status, _ = pf.rerank(ep, shortlist, "openai", {"model": "bad-cache-test"}, ask=lambda *_: "not json", retries=1)
        self.assertEqual(status, "fallback")
        self.assertEqual(len(pf._RESPONSE_CACHE), 0)

    def test_prompt_response_cache(self):
        ep = next(ep for ep in self.episodes if ep["number"] == 1)
        shortlist = pf.thin(pf.build(ep, {}, 1.5), lambda c: c["score"], 1)
        pf._RESPONSE_CACHE.clear()
        response = json.dumps([{"id": 0, "score": 8, "hook": "hook", "cliffhanger": "cliff",
                                "why": "reason", "standalone_ok": True}])
        ask = Mock(return_value=response)
        _, first_status, _ = pf.rerank(ep, shortlist, "openai", {"model": "cache-test"}, ask=ask)
        _, second_status, _ = pf.rerank(ep, shortlist, "openai", {"model": "cache-test"}, ask=ask)
        self.assertEqual(first_status, "ok")
        self.assertEqual(second_status, "cache")
        ask.assert_called_once()

    def test_pipeline_output_constraints(self):
        promos, meta = pf.run_pipeline(self.episodes, return_meta=True)
        self.assertEqual(meta["episodes"], len(self.episodes))
        self.assertTrue(promos)
        self.assertLessEqual(max(sum(1 for p in promos if p["episode"] == ep) for ep in range(1, 51)), 2)
        for promo in promos:
            self.assertTrue(pf.valid(promo, next(ep["length"] for ep in self.episodes if ep["number"] == promo["episode"])))
            self.assertGreaterEqual(promo["score"], 0)
            self.assertLessEqual(promo["score"], 10)

    def test_top_per_episode_is_capped_at_two(self):
        with self.assertRaises(ValueError):
            pf.run_pipeline(self.episodes, top_per_ep=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
