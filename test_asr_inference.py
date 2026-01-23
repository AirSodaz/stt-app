
import unittest
import re
import sys
import os

# --- Start of copied code from asr_inference.py for standalone testing ---

# This allows the test to run without needing the full asr_inference module and its heavy dependencies.
# In a real-world scenario, this would be part of a proper test suite structure.

# SenseVoice emoji mappings (official)
SENSEVOICE_EMOJI_DICT = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

# Sort keys by length descending to handle multi-tag patterns first
sorted_tags = sorted(SENSEVOICE_EMOJI_DICT.keys(), key=lambda k: -len(k))

# Create a single regex for all tags. This is more efficient than iterating.
SENSEVOICE_TAG_PATTERN = re.compile("|".join(re.escape(tag) for tag in sorted_tags))


def sensevoice_postprocess(text: str, show_emoji: bool = True) -> str:
    """
    Post-process SenseVoice output text.
    """
    # First normalize tags with spaces: "< | zh | >" -> "<|zh|>"
    text = re.sub(r'<\s*\|\s*([^|]*?)\s*\|\s*>', lambda m: f'<|{m.group(1).strip()}|>', text)

    def replace_tag(match):
        """Replacer function for re.sub."""
        tag = match.group(0)
        if show_emoji:
            return SENSEVOICE_EMOJI_DICT.get(tag, "")
        return "" # If not showing emoji, remove all known tags

    # Use the pre-compiled regex to replace all known tags in one pass
    text = SENSEVOICE_TAG_PATTERN.sub(replace_tag, text)

    # Remove any remaining unrecognized tags
    text = re.sub(r'<\|[^|]*\|>', '', text)

    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- End of copied code ---


class TestSenseVoicePostprocess(unittest.TestCase):

    def test_basic_emoji_replacement(self):
        """Test that basic tags are replaced with emojis."""
        input_text = "Hello <|HAPPY|> world <|Laughter|>"
        expected_text = "Hello 😊 world 😀"
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)

    def test_tag_removal_without_emoji(self):
        """Test that tags are removed when show_emoji is False."""
        input_text = "Hello <|HAPPY|> world <|Laughter|>"
        expected_text = "Hello world"
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=False), expected_text)

    def test_multi_tag_pattern(self):
        """Test that longer, multi-part tags are correctly identified."""
        input_text = "<|nospeech|><|Event_UNK|>"
        expected_text = "❓"
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)

    def test_tags_with_spaces(self):
        """Test that tags with inconsistent spacing are normalized and replaced."""
        input_text = "This is <| SAD |>"
        expected_text = "This is 😔"
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)

    def test_unrecognized_tags(self):
        """Test that unrecognized tags are removed."""
        input_text = "This is an <|UNKNOWNTAG|>."
        expected_text = "This is an ."
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)

    def test_empty_input(self):
        """Test that an empty string is handled correctly."""
        input_text = ""
        expected_text = ""
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)

    def test_no_tags_in_input(self):
        """Test that a string with no tags remains unchanged."""
        input_text = "This is a normal sentence."
        expected_text = "This is a normal sentence."
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)

    def test_mixed_known_and_unknown_tags(self):
        """Test a mix of known and unknown tags."""
        input_text = "A <|HAPPY|> face and an <|ALIEN|> artifact."
        expected_text = "A 😊 face and an artifact."
        self.assertEqual(sensevoice_postprocess(input_text, show_emoji=True), expected_text)


if __name__ == '__main__':
    # Ensure the output is encoded in UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    unittest.main()
