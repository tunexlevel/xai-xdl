import unittest
from unittest.mock import patch

from helper.utils import strip_atom_mapping_labels
from predict import api_prediction
from predict.api_prediction import _trim_special_attention_rows


class ForwardPredictionAlignmentTests(unittest.TestCase):
    def test_attention_rows_keep_sos_prediction_and_drop_trailing_special_rows(self):
        attention = [[index] for index in range(22)]

        aligned = _trim_special_attention_rows(attention, target_length=20)

        self.assertEqual(len(aligned), 20)
        self.assertEqual(aligned[0], [0])
        self.assertEqual(aligned[-1], [19])

    def test_map_labels_are_removed_from_token_text_without_changing_count(self):
        mapped_tokens = ["[CH3:1]", "[O:2]", "c", "1"]

        unmapped_tokens = [
            strip_atom_mapping_labels(token) for token in mapped_tokens
        ]

        self.assertEqual(unmapped_tokens, ["[CH3]", "[O]", "c", "1"])
        self.assertEqual(len(unmapped_tokens), len(mapped_tokens))
        self.assertFalse(any(":" in token for token in unmapped_tokens))

    def test_forward_response_has_unmapped_labels_and_aligned_attention(self):
        bundle = {
            "model": object(),
            "token2idx": {
                "<pad>": 0,
                "<sos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "[CH3:1]": 4,
                "[OH:2]": 5,
                "[CH3:7]": 6,
                "[OH:8]": 7,
            },
            "idx2token": {
                6: "[CH3:7]",
                7: "[OH:8]",
            },
            "pad_idx": 0,
            "sos_idx": 1,
            "eos_idx": 2,
            "unk_idx": 3,
            "mapped": True,
        }
        candidate = (
            [1, 6, 7, 2],
            -0.5,
            [[0.1], [0.2], [0.3], [0.4]],
        )

        with (
            patch.object(api_prediction, "_load_model", return_value=bundle),
            patch.object(
                api_prediction,
                "_run_beam_search",
                return_value=[candidate],
            ) as beam_search,
        ):
            result = api_prediction.predict_product(
                "[CH3:1][OH:2]",
                top_k=1,
            )[0]

        self.assertEqual(beam_search.call_args.args[1].tolist(), [[4, 5]])
        self.assertFalse(":" in result["prediction"])
        self.assertEqual(result["source_tokens"], ["C", "O"])
        self.assertEqual(result["target_tokens"], ["C", "O"])
        self.assertEqual(
            result["attention_weights"],
            [[0.1], [0.2]],
        )


if __name__ == "__main__":
    unittest.main()
