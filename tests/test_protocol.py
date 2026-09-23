import unittest
from xml.etree import ElementTree

from protocol.index import (
    protocol_to_xdl,
    reaction_to_draft_protocol,
    recipe_to_protocol,
)


SAMPLE_RECIPE = """
add aniline 1.0ml to vial 1
wash vial 1 with water 2ml x2
stir vial 1, vial 2 for 10s at medium
capture dark spectrum
capture reference spectrum vial 24
measure spectrum vial 3
"""


class ProtocolGenerationTests(unittest.TestCase):
    def test_sample_recipe_is_normalized(self):
        protocol = recipe_to_protocol(SAMPLE_RECIPE)

        self.assertEqual(len(protocol), 6)
        self.assertEqual(protocol[0]["action"], "transfer")
        self.assertEqual(protocol[0]["reagent"], "Aniline")
        self.assertEqual(protocol[0]["volume_ml"], 1.0)
        self.assertEqual(protocol[0]["destination"], "vial_1")
        self.assertEqual(protocol[1]["cycles"], 2)
        self.assertEqual(protocol[2]["vials"], ["vial_1", "vial_2"])
        self.assertEqual(protocol[4]["vial"], "vial_24")
        self.assertEqual(protocol[5]["vial"], "vial_3")

    def test_sample_recipe_compiles_to_well_formed_xml(self):
        protocol = recipe_to_protocol(SAMPLE_RECIPE)
        xdl = protocol_to_xdl(protocol, run_name="test_run")
        root = ElementTree.fromstring(xdl)

        self.assertEqual(root.tag, "XDL")
        self.assertIn('<Add vessel="vial_1" reagent="Aniline" volume="1.0 mL" />', xdl)
        self.assertIn('<CleanVessel vessel="vial_1" solvent="Water" volume="2.0 mL" repeats="2" />', xdl)
        self.assertIn('<MeasureSpectrum vial="vial_3"', xdl)

    def test_empty_recipe_is_rejected(self):
        with self.assertRaises(ValueError):
            recipe_to_protocol("\n")

    def test_missing_volume_is_rejected(self):
        with self.assertRaises(ValueError):
            recipe_to_protocol("add aniline to vial 1")

    def test_vial_outside_platform_is_rejected(self):
        with self.assertRaises(ValueError):
            recipe_to_protocol("add aniline 1ml to vial 25")

    def test_stir_vial_outside_platform_is_rejected(self):
        with self.assertRaises(ValueError):
            recipe_to_protocol("stir vial 1, vial 25 for 10s at medium")

    def test_unknown_instruction_is_rejected(self):
        with self.assertRaises(ValueError):
            recipe_to_protocol("perform an unknown operation")

    def test_reaction_smiles_creates_reviewable_draft(self):
        steps, warnings = reaction_to_draft_protocol(
            "Brc1ccc(Br)nc1.CN(C)C=O",
            "O=Cc1ccc(Br)cn1",
        )
        xdl = protocol_to_xdl(steps, run_name="draft_test")

        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0]["reagent"], "Brc1ccc(Br)nc1")
        self.assertEqual(steps[1]["reagent"], "CN(C)C=O")
        self.assertEqual(steps[2]["action"], "timed_stir")
        self.assertEqual(steps[3]["action"], "measure_spectrum")
        self.assertTrue(any("SMILES" in warning for warning in warnings))
        ElementTree.fromstring(xdl)
        self.assertIn("Brc1ccc(Br)nc1", xdl)


if __name__ == "__main__":
    unittest.main()
