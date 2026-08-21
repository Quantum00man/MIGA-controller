import unittest

from app.core.sequence_markers import (
    add_sequence_marker,
    inspect_sequence_markers,
    marked_filename,
    marker_definitions_for_sequence,
    normalize_marker_profiles,
    sequence_marker_profile_key,
    normalize_marker_definitions,
    remove_sequence_marker,
    render_auto_marker_sequence,
    update_sequence_marker,
)


MOT = """# header
+2us AOM_Raman1 =1                    (23)
+330995.2us DDS1 [382]                (2)
+80us TTL_AOM_Raman1 = ON             (49)
+5026us AOM_Det= OFF                  (52)
"""


def definition(marker_id, kind, **overrides):
    defaults = {
        "id": marker_id,
        "display_name": marker_id.replace("_", " "),
        "kind": kind,
        "decimals": 3 if kind == "dac_value" else 0,
        "hard_min": 0,
        "hard_max": 1000,
        "default_start": 1,
        "default_stop": 2,
        "default_step": 1,
        "default_method": "step_size",
        "expected_command": "",
        "expected_channel": "",
        "has_compensation": False,
    }
    defaults.update(overrides)
    return defaults


class MarkerInspectionTests(unittest.TestCase):
    def test_detects_duration_dds_and_dac_candidates(self):
        inspection = inspect_sequence_markers(MOT)
        candidates = {
            candidate["candidate_id"]
            for line in inspection["lines"]
            for candidate in line["candidates"]
        }
        self.assertIn("2:dac_value", candidates)
        self.assertIn("3:dds_element", candidates)
        self.assertIn("4:duration", candidates)

    def test_add_and_remove_duration_pair(self):
        marked = add_sequence_marker(MOT, "Raman pulse", 4, "duration", 5)
        self.assertIn("###SCAN:RAMAN_PULSE###", marked)
        self.assertIn("###COMP:RAMAN_PULSE###", marked)
        inspection = inspect_sequence_markers(
            marked,
            [definition("RAMAN_PULSE", "duration", has_compensation=True)],
        )
        self.assertEqual([item["status"] for item in inspection["markers"]], ["defined", "defined"])
        self.assertEqual(remove_sequence_marker(marked, "RAMAN_PULSE"), MOT)

    def test_existing_unknown_marker_is_reported_undefined(self):
        marked = "###SCAN:REPUMP_DAC###\n" + MOT
        marker = inspect_sequence_markers(marked)["markers"][0]
        self.assertEqual(marker["status"], "undefined")

    def test_definition_conflict_reports_expected_command(self):
        marked = add_sequence_marker(MOT, "REPUMP_DAC", 2, "dac_value")
        marker = inspect_sequence_markers(
            marked,
            [definition("REPUMP_DAC", "dac_value", expected_command="AOM_Det")],
        )["markers"][0]
        self.assertEqual(marker["status"], "conflict")

    def test_marked_filename_does_not_duplicate_suffix(self):
        self.assertEqual(marked_filename("sequence.mot"), "sequence_marked.mot")
        self.assertEqual(marked_filename("sequence_marked.mot"), "sequence_marked.mot")



class MarkerCompensationInspectionTests(unittest.TestCase):
    def test_required_compensation_missing_is_conflict(self):
        marked = add_sequence_marker(MOT, "LABEL_DURATION", 4, "duration")
        inspection = inspect_sequence_markers(
            marked,
            [definition("LABEL_DURATION", "duration", hard_min=1, hard_max=5000, has_compensation=True)],
        )
        marker = next(item for item in inspection["markers"] if item["role"] == "scan")
        self.assertEqual(marker["status"], "conflict")
        self.assertIn("requires a compensation", marker["message"])

    def test_unexpected_compensation_is_conflict(self):
        marked = add_sequence_marker(MOT, "LABEL_DURATION", 4, "duration", 5)
        inspection = inspect_sequence_markers(
            marked,
            [definition("LABEL_DURATION", "duration", hard_min=1, hard_max=5000, has_compensation=False)],
        )
        marker = next(item for item in inspection["markers"] if item["role"] == "scan")
        self.assertEqual(marker["status"], "conflict")
        self.assertIn("disabled", marker["message"])


class MarkerUpdateTests(unittest.TestCase):
    def test_renames_and_moves_marker_to_different_kind(self):
        marked = add_sequence_marker(MOT, "OLD_DURATION", 4, "duration", 5)
        updated = update_sequence_marker(
            marked,
            "OLD_DURATION",
            "NEW_DETUNING",
            3,
            "dds_element",
        )
        inspection = inspect_sequence_markers(updated)
        scan = next(item for item in inspection["markers"] if item["role"] == "scan")
        self.assertEqual(scan["id"], "NEW_DETUNING")
        self.assertEqual(scan["kind"], "dds_element")
        self.assertFalse(any(item["role"] == "comp" for item in inspection["markers"]))
        self.assertNotIn("OLD_DURATION", updated)

    def test_changes_compensation_target(self):
        source = MOT + "+900us WAIT= OFF                    (77)\n"
        marked = add_sequence_marker(source, "DURATION", 4, "duration", 5)
        current = inspect_sequence_markers(marked)
        scan_line = next(item["target_line_number"] for item in current["markers"] if item["role"] == "scan")
        new_comp_line = next(
            line["line_number"] for line in current["lines"]
            if "+900us WAIT" in line["source"]
        )
        updated = update_sequence_marker(marked, "DURATION", "DURATION", scan_line, "duration", new_comp_line)
        inspection = inspect_sequence_markers(updated)
        compensation = next(item for item in inspection["markers"] if item["role"] == "comp")
        self.assertIn("+900us WAIT", compensation["target_source"])

    def test_inspection_attaches_marker_metadata_to_target_line(self):
        marked = add_sequence_marker(MOT, "REPUMP", 2, "dac_value")
        line = next(item for item in inspect_sequence_markers(marked)["lines"] if item["marked"])
        self.assertEqual(line["markers"][0]["id"], "REPUMP")
        self.assertEqual(line["markers"][0]["role"], "scan")

class MarkerRenderingTests(unittest.TestCase):
    def test_renders_three_marker_types_with_required_formats(self):
        marked = add_sequence_marker(MOT, "REPUMP_DAC", 2, "dac_value")
        marked = add_sequence_marker(marked, "LABEL_DETUNING", 4, "dds_element")
        # The second insertion shifted the original duration line from 4 to 6.
        marked = add_sequence_marker(marked, "LABEL_DURATION", 6, "duration", 7)
        definitions = [
            definition("REPUMP_DAC", "dac_value", hard_min=0, hard_max=1, default_start=0, default_stop=1, default_step=0.1),
            definition("LABEL_DETUNING", "dds_element", hard_min=0, hard_max=1023),
            definition("LABEL_DURATION", "duration", hard_min=1, hard_max=5000, has_compensation=True),
        ]
        rendered = render_auto_marker_sequence(
            marked,
            ["REPUMP_DAC", "LABEL_DETUNING", "LABEL_DURATION"],
            [0.375, 322, 100],
            definitions,
        )
        self.assertIn("AOM_Raman1 =0.375", rendered)
        self.assertIn("DDS1 [322]", rendered)
        self.assertIn("+100us TTL_AOM_Raman1", rendered)
        self.assertIn("+5006us AOM_Det", rendered)

    def test_rejects_value_outside_individual_hard_limits(self):
        marked = add_sequence_marker(MOT, "REPUMP_DAC", 2, "dac_value")
        with self.assertRaisesRegex(ValueError, "outside hard limits"):
            render_auto_marker_sequence(
                marked,
                ["REPUMP_DAC"],
                [1.1],
                [definition("REPUMP_DAC", "dac_value", hard_min=0, hard_max=1, default_start=0, default_stop=1, default_step=0.1)],
            )

    def test_rejects_zero_or_negative_compensation(self):
        marked = add_sequence_marker(MOT, "LABEL_DURATION", 4, "duration", 5)
        with self.assertRaisesRegex(ValueError, "compensation must be greater than 0"):
            render_auto_marker_sequence(
                marked,
                ["LABEL_DURATION"],
                [5106],
                [definition(
                    "LABEL_DURATION",
                    "duration",
                    hard_min=1,
                    hard_max=6000,
                    default_stop=5000,
                    has_compensation=True,
                )],
            )

    def test_integer_marker_rejects_fractional_value(self):
        marked = add_sequence_marker(MOT, "LABEL_DETUNING", 3, "dds_element")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            render_auto_marker_sequence(
                marked,
                ["LABEL_DETUNING"],
                [322.5],
                [definition("LABEL_DETUNING", "dds_element", hard_min=0, hard_max=1023)],
            )



class MarkerProfileTests(unittest.TestCase):
    def test_same_marker_id_can_have_independent_sequence_definitions(self):
        profiles = normalize_marker_profiles({
            "sequence_a.mot": [definition("SHARED_NAME", "dac_value", hard_min=0, hard_max=1, default_stop=1)],
            "sequence_b.mot": [definition("SHARED_NAME", "dac_value", hard_min=-3, hard_max=3, default_start=-1, default_stop=2)],
        }, strict=True)
        settings = {"sequence_marker_profiles": profiles, "sequence_marker_definitions": []}
        first = marker_definitions_for_sequence(settings, "sequence_a_marked.mot")[0]
        second = marker_definitions_for_sequence(settings, "sequence_b.mot")[0]
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["hard_max"], 1)
        self.assertEqual(second["hard_max"], 3)

    def test_marked_suffix_uses_same_profile(self):
        self.assertEqual(
            sequence_marker_profile_key("Experiment_marked.mot"),
            sequence_marker_profile_key("Experiment.mot"),
        )

    def test_legacy_definitions_are_used_only_without_profile(self):
        legacy = definition("LEGACY", "dds_element", hard_min=0, hard_max=1023)
        scoped = definition("SCOPED", "duration", hard_min=1, hard_max=100)
        settings = {
            "sequence_marker_definitions": [legacy],
            "sequence_marker_profiles": {"new sequence": [scoped]},
        }
        self.assertEqual(marker_definitions_for_sequence(settings, "old.mot")[0]["id"], "LEGACY")
        self.assertEqual(marker_definitions_for_sequence(settings, "new sequence.mot")[0]["id"], "SCOPED")

class MarkerDefinitionTests(unittest.TestCase):
    def test_rejects_defaults_outside_hard_limits(self):
        with self.assertRaisesRegex(ValueError, "exceeds hard limits"):
            normalize_marker_definitions(
                [definition("DAC", "dac_value", hard_min=0, hard_max=1, default_stop=2)],
                strict=True,
            )

    def test_rejects_fractional_point_count(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            normalize_marker_definitions(
                [definition("DAC", "dac_value", default_method="n_points", default_step=2.5)],
                strict=True,
            )

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            normalize_marker_definitions(
                [definition("DAC", "dac_value"), definition("DAC", "dac_value")],
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
