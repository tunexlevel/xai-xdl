import unittest

from helper.utils import tokenize_smiles
from predict.api_prediction_retro import (
    _remove_atom_mapping_preserve_order,
    _trim_special_attention_rows,
)


class RetroPredictionAlignmentTests(unittest.TestCase):
    def test_attention_rows_start_at_sos_prediction_and_match_target_length(self):
        attention = [[index] for index in range(22)]

        aligned = _trim_special_attention_rows(attention, target_length=20)

        self.assertEqual(len(aligned), 20)
        self.assertEqual(aligned[0], [0])
        self.assertEqual(aligned[-1], [19])

    def test_mapping_removal_preserves_decoded_token_order(self):
        decoded_smiles = "[CH3:1][O:2]c1ccccc1"

        unmapped_smiles = _remove_atom_mapping_preserve_order(decoded_smiles)

        self.assertEqual(unmapped_smiles, "[CH3][O]c1ccccc1")
        self.assertEqual(
            tokenize_smiles(unmapped_smiles),
            ["[CH3]", "[O]", "c", "1", "c", "c", "c", "c", "c", "1"],
        )


if __name__ == "__main__":
    unittest.main()