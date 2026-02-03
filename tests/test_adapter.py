import sys
import os
import unittest

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.adapter import ActionAdapter

class TestActionAdapter(unittest.TestCase):
    
    def test_valid_marketing(self):
        raw = {"type": "marketing", "params": {"amount": "5000", "extra": "ignore"}}
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["type"], "marketing")
        self.assertEqual(clean["params"]["amount"], 5000.0)
        self.assertIsInstance(clean["params"]["amount"], float)

    def test_invalid_type_fallback(self):
        raw = {"type": "destroy_company", "params": {}}
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["type"], "skip")
        
    def test_malformed_numbers(self):
        # String number -> Float
        raw = {"type": "pricing", "params": {"price": "19.99"}}
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["params"]["price"], 19.99)
        
        # Invalid number -> Skip/Default? 
        # Verify it logs an error but doesn't crash
        raw_bad = {"type": "pricing", "params": {"price": "free"}}
        with self.assertLogs("AgentAdapter", level='ERROR') as cm:
            clean_bad = ActionAdapter.translate_action(raw_bad)
        
        self.assertEqual(clean_bad["type"], "skip")
        self.assertTrue(any("Parameter validation failed" in o for o in cm.output))

    def test_negative_values_clamped(self):
        raw = {"type": "hiring", "params": {"count": -5}}
        clean = ActionAdapter.translate_action(raw)
        self.assertEqual(clean["params"]["count"], 0)

if __name__ == "__main__":
    unittest.main()
