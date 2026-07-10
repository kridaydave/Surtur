import unittest
import sys
import os

# Add src to sys.path if not present
sys.path.insert(0, os.path.dirname(__file__))

from rewards import (
    format_reward_fn,
    math_correctness_reward_fn,
    p_grpo_format_reward_fn,
    step_grpo_reward_fn,
    tag_spam_penalty_fn
)
from eval_harness import get_choice_token_id

class MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        # Simplistic mock tokenization
        # Split by space, add dummy start token if add_special_tokens
        tokens = []
        if add_special_tokens:
            tokens.append(101)  # [START]
        # Simulate simple BPE merging of trailing space with letter
        if text.endswith(" Answer: A"):
            tokens.extend([200, 300, 401]) # 401 is token for " A"
        elif text.endswith(" Answer: B"):
            tokens.extend([200, 300, 402])
        elif text.endswith(" Answer: "):
            tokens.extend([200, 300, 400]) # 400 is token for " "
        else:
            for word in text.split():
                tokens.append(hash(word) % 1000 + 500)
        return tokens

    def convert_tokens_to_ids(self, token):
        return hash(token) % 1000 + 500

class TestSurturRewardsAndHarness(unittest.TestCase):
    def test_format_reward(self):
        # 1. Perfect single-block format -> 1.0
        comp1 = "<think> some thinking </think> Final Answer: 42"
        self.assertEqual(format_reward_fn([], [comp1])[0], 1.0)
        
        # 2. Unclosed block -> 0.2
        comp2 = "<think> unclosed thinking"
        self.assertEqual(format_reward_fn([], [comp2])[0], 0.2)
        
        # 3. No tags -> 0.0
        comp3 = "No structure answer"
        self.assertEqual(format_reward_fn([], [comp3])[0], 0.0)

    def test_math_correctness_reward(self):
        # Equivalent
        self.assertEqual(math_correctness_reward_fn([], ["The answer is \\boxed{42}"], ["42.0"])[0], 1.0)
        self.assertEqual(math_correctness_reward_fn([], ["The answer is 1,000"], ["1000"])[0], 1.0)
        
        # Incorrect
        self.assertEqual(math_correctness_reward_fn([], ["The answer is 10"], ["42"])[0], 0.0)

    def test_tag_spam_penalty(self):
        # No bad tags
        self.assertEqual(tag_spam_penalty_fn([], ["<think> valid </think> Answer: 42"])[0], 0.0)
        
        # Bad tags -> -0.3 penalty
        self.assertEqual(tag_spam_penalty_fn([], ["<think> valid </think> <step1> bad </step1>"])[0], -0.6)

    def test_get_choice_token_id(self):
        tokenizer = MockTokenizer()
        prompt = "Question: what is 2+2? Answer: "
        
        # If tokenizer merges, it should fall back to encoding letter alone
        token_id_a = get_choice_token_id(tokenizer, prompt, "A")
        # In mock, encode(prompt + "A") has same length as encode(prompt).
        # So it falls back to tokenizer.encode("A", add_special_tokens=False)
        self.assertIsNotNone(token_id_a)

if __name__ == "__main__":
    unittest.main()
