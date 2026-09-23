import unittest

from explain import explain_prediction


class ExplanationTests(unittest.TestCase):
    def test_reports_formylation_cue_and_attention(self):
        result = explain_prediction(
            "Brc1ccc(Br)nc1.CN(C)C=O",
            "O=Cc1ccc(Br)cn1",
            source_tokens=["Br", "c", ".", "C", "N", "C", "C", "=", "O"],
            target_tokens=["O", "=", "C", "c"],
            attention_weights=[
                [0.01, 0.01, 0.01, 0.10, 0.05, 0.05, 0.05, 0.20, 0.53],
                [0.40, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.50, 0.15],
                [0.10, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05],
            ],
        )

        self.assertEqual(result["reaction_class"], "aryl formylation")
        self.assertEqual(result["reaction_class_confidence"], "low")
        self.assertTrue(result["attention_summary"]["available"])
        self.assertEqual(result["attention_summary"]["orientation"], "target_token_to_source_token")
        top_tokens = {
            item["token"]
            for item in result["attention_summary"]["top_source_tokens"]
        }
        self.assertTrue({"=", "O"}.intersection(top_tokens))
        self.assertIn("attended", result["explanation"])
        self.assertIn("reactants", result["llm_input"])

    def test_handles_missing_attention(self):
        result = explain_prediction("CCO", "CC=O")

        self.assertFalse(result["attention_summary"]["available"])
        self.assertEqual(result["reaction_class"], "unclassified transformation")


if __name__ == "__main__":
    unittest.main()
