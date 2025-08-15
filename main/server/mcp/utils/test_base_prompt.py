# main/server/mcp/utils/test_base_prompt.py
import unittest
from .base_prompt import BasePrompt

class TestBasePrompt(unittest.TestCase):
    def setUp(self):
        self.base_prompt = BasePrompt()

    def test_format_prompt_default(self):
        prompt = self.base_prompt.format_prompt("Hello, world!")
        self.assertEqual(prompt["user"], "Hello, world!")
        self.assertIn("You are Grok, created by xAI", prompt["system"])
        self.assertNotIn("context", prompt)

    def test_format_prompt_with_context(self):
        context = {"key": "value"}
        prompt = self.base_prompt.format_prompt("Hello, world!", context=context)
        self.assertEqual(prompt["user"], "Hello, world!")
        self.assertEqual(prompt["context"], context)

    def test_format_prompt_custom_system(self):
        system_prompt = "Custom system prompt"
        prompt = self.base_prompt.format_prompt("Hello, world!", system_prompt=system_prompt)
        self.assertEqual(prompt["user"], "Hello, world!")
        self.assertEqual(prompt["system"], system_prompt)

    def test_validate_prompt_valid(self):
        prompt = {"user": "Hello, world!", "system": "Custom system prompt"}
        result = self.base_prompt.validate_prompt(prompt)
        self.assertTrue(result)

    def test_validate_prompt_missing_user(self):
        prompt = {"system": "Custom system prompt"}
        result = self.base_prompt.validate_prompt(prompt)
        self.assertFalse(result)

    def test_validate_prompt_empty_user(self):
        prompt = {"user": "", "system": "Custom system prompt"}
        result = self.base_prompt.validate_prompt(prompt)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
