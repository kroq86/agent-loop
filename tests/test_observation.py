"""An observation over the budget keeps whole result items and says what it left out."""
import json
import unittest

from agent_loop.loop import fit_observation


class ObservationBudget(unittest.TestCase):
    def payload(self, n, size=300):
        return json.dumps({"tool": "search", "result": [{"id": f"d{i}", "text": f"fact {i} " + "x" * size}
                                                        for i in range(n)]})

    def test_small_payload_is_unchanged(self):
        p = self.payload(2)
        self.assertEqual(fit_observation(p, limit=10_000), (p, None))

    def test_json_result_keeps_whole_items_and_counts_the_rest(self):
        shown, cut = fit_observation(self.payload(10), limit=1500)
        data = json.loads(shown)
        self.assertLessEqual(len(shown), 1500)
        self.assertEqual(len(data["result"]) + data["omitted_results"], 10)
        self.assertTrue(all(item["text"].endswith("x" * 300) for item in data["result"]))
        self.assertEqual(cut["omitted_items"], data["omitted_results"])

    def test_other_payload_is_cut_with_a_visible_note(self):
        shown, cut = fit_observation("y" * 5000, limit=1000)
        self.assertTrue(shown.startswith("y" * 1000))
        self.assertIn("1000 of 5000 characters", shown)
        self.assertEqual(cut["payload_chars"], 5000)


if __name__ == "__main__":
    unittest.main()
