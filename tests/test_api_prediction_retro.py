import unittest

from helper.utils import tokenize_smiles
from predict.api_prediction_retro import (
    _remove_atom_mapping_labels,
    _prepare_display_smiles,
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

        unmapped_smiles = _remove_atom_mapping_labels(decoded_smiles)

        self.assertEqual(unmapped_smiles, "[CH3][O]c1ccccc1")
        self.assertEqual(
            tokenize_smiles(unmapped_smiles),
            ["[CH3]", "[O]", "c", "1", "c", "c", "c", "c", "c", "1"],
        )

    def test_display_smiles_removes_redundant_brackets_and_hydrogens(self):
        decoded_smiles = (
            "[C](O)(=[O])[CH2][c]1[cH][cH][cH][cH][cH]1."
            "[CH3][c]1[n][c]([NH2])[s][c]1[C](=[O])[NH][CH2][c]1"
            "[cH][cH][cH][cH][cH]1"
        )

        display_smiles, target_tokens = _prepare_display_smiles(decoded_smiles)

        self.assertEqual(
            display_smiles,
            "C(O)(=O)Cc1ccccc1.Cc1nc(N)sc1C(=O)NCc1ccccc1",
        )
        self.assertEqual(len(target_tokens), len(tokenize_smiles(decoded_smiles)))
        self.assertNotIn("[", display_smiles)
        self.assertFalse(any("[" in token for token in target_tokens))


if __name__ == "__main__":
    unittest.main()