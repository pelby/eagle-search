"""R2/R3 contract tests for the fixed, vision-only caption prompt."""

from __future__ import annotations

import json
import unittest

from src.captioning.prompt import CAPTION_PROMPT_VERSION, caption_prompt, output_schema


class CaptionSchemaTests(unittest.TestCase):
    def test_prompt_is_versioned_vision_only_and_does_not_leak_metadata(self) -> None:
        prompt = caption_prompt()

        self.assertEqual(CAPTION_PROMPT_VERSION, "caption-v3")
        self.assertIn("visible evidence", prompt.casefold())
        self.assertIn("do not infer", prompt.casefold())
        self.assertIn("retrieval aliases, not factual claims", prompt.casefold())
        self.assertIn("classroom", prompt.casefold())
        for forbidden in ("filename", "tags", "human notes", "generation prompt"):
            self.assertNotIn(forbidden, prompt.casefold())

    def test_frozen_schema_is_strict_and_matches_caption_result_contract(self) -> None:
        schema = output_schema()

        self.assertEqual(schema["title"], "CaptionResultV1")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            schema["properties"]["contract_version"],
            {"type": "integer", "const": 1},
        )
        self.assertFalse(schema["properties"]["visible_text"]["items"]["additionalProperties"])
        self.assertEqual(json.loads(json.dumps(schema)), schema)


if __name__ == "__main__":
    unittest.main()
