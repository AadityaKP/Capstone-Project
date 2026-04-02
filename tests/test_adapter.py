import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.adapter import ActionAdapter

class TestActionAdapter(unittest.TestCase):
    
    def test_valid_bundle_is_sanitized(self):
        raw = {
            "marketing": {"spend": "5000", "channel": "brand"},
            "hiring": {"hires": "2", "cost_per_employee": "9000"},
            "product": {"r_and_d_spend": "2500"},
            "pricing": {"price_change_pct": "0.15"},
        }
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["marketing"]["spend"], 5000.0)
        self.assertEqual(clean["marketing"]["channel"], "brand")
        self.assertEqual(clean["hiring"]["hires"], 2)
        self.assertEqual(clean["hiring"]["cost_per_employee"], 9000.0)
        self.assertEqual(clean["product"]["r_and_d_spend"], 2500.0)
        self.assertEqual(clean["pricing"]["price_change_pct"], 0.15)

    def test_non_dict_returns_noop(self):
        raw = "bad input"
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean, ActionAdapter._get_noop())
        
    def test_negative_values_clamped(self):
        raw = {
            "marketing": {"spend": -100.0, "channel": "email"},
            "hiring": {"hires": -5, "cost_per_employee": -1},
            "product": {"r_and_d_spend": -200},
            "pricing": {"price_change_pct": 5.0},
        }
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["marketing"]["spend"], 0.0)
        self.assertEqual(clean["marketing"]["channel"], "ppc")
        self.assertEqual(clean["hiring"]["hires"], 0)
        self.assertEqual(clean["hiring"]["cost_per_employee"], 1.0)
        self.assertEqual(clean["product"]["r_and_d_spend"], 0.0)
        self.assertEqual(clean["pricing"]["price_change_pct"], 1.0)

    def test_invalid_subsections_fallback_independently(self):
        raw = {
            "marketing": {"spend": "broken", "channel": "ppc"},
            "hiring": {"hires": 1, "cost_per_employee": 10000},
        }
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["marketing"], {"spend": 0.0, "channel": "ppc"})
        self.assertEqual(clean["hiring"]["hires"], 1)
        self.assertEqual(clean["product"]["r_and_d_spend"], 0.0)
        self.assertEqual(clean["pricing"]["price_change_pct"], 0.0)

if __name__ == "__main__":
    unittest.main()
