import re


def get_completion_text(comp) -> str:
    """
    Safely extracts the generated text string from a completion.
    If the completion is a list of message dicts (from a chat template),
    it retrieves the content of the assistant's message.
    """
    if isinstance(comp, list):
        for msg in comp:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # Fallback to the last message's content
        if len(comp) > 0 and isinstance(comp[-1], dict):
            return comp[-1].get("content", "")
    elif isinstance(comp, str):
        return comp
    return ""


def extract_xml_answer(text: str) -> str:
    """
    Extracts the final answer from the model completion.
    Looks after the </think> tag if present, and extracts either:
    1. A number wrapped in \boxed{...}
    2. The last number found in the text.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Remove commas from numbers to avoid matching errors (e.g., 1,000 -> 1000)
    text_clean = text.replace(",", "")

    # 1. Look for \boxed{...}
    boxed_match = re.search(r"\\boxed\{((?:[^{}]+|\{(?:[^{}]+|\{[^{}]*\})*\})*)\}", text_clean)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 2. Look for the last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        return numbers[-1]

    return ""


def is_mathematically_equivalent(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    # Clean commas to prevent float conversion crashes
    s1_clean = s1.replace(",", "").strip()
    s2_clean = s2.replace(",", "").strip()
    try:
        return float(s1_clean) == float(s2_clean)
    except ValueError:
        return s1_clean.lower() == s2_clean.lower()


def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """
    Standard formatting reward. Checks if the model output follows the <think>...</think> structure.
    Returns:
        1.0 for perfect single-block structure.
        0.3 for multi-block formats (exploit prevention).
        0.2 for unclosed blocks.
        0.5 for malformed/empty trailing.
        0.0 for no structure.
    """
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)

        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")

        # Strict single-block format check
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            # Ensure correct ordering and some content after </think>
            if start_idx < end_idx and len(comp_text[end_idx + 8 :].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif num_start > 1 or num_end > 1:
            # Multi-block formatting penalty
            rewards.append(0.3)
        elif num_start == 1 and num_end == 0:
            # Unclosed block penalty
            rewards.append(0.2)
        elif num_start == 0 and num_end == 1:
            # Missing start tag but has end tag
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def math_correctness_reward_fn(
    prompts, completions, target_answer, **kwargs
) -> list[float]:
    """
    Standard math correctness reward.
    Compares the extracted numerical answer with the target ground truth.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")

        if extracted and is_mathematically_equivalent(extracted, target_clean):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def p_grpo_format_reward_fn(
    prompts, completions, target_answer, **kwargs
) -> list[float]:
    """
    Posterior-GRPO Formatting Reward.
    Zeroes out the formatting (process) reward if the final math answer is incorrect.
    This prevents the model from being reinforced for 'looking smart' while getting the math wrong.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)

        # Check correctness first
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)

        if not is_correct:
            rewards.append(0.0)
            continue

        # If correct, evaluate format strictly
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")

        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8 :].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif num_start > 1 or num_end > 1:
            rewards.append(0.3)
        elif num_start == 1 and num_end == 0:
            rewards.append(0.2)
        elif num_start == 0 and num_end == 1:
            rewards.append(0.2)
        else:
            rewards.append(0.0)

    return rewards


def step_grpo_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Step-GRPO Decaying Reward based on Monologue Word Count to prevent vocabulary evasion.
    Applies a 100-word grace window to give the model "thinking space" for complex math,
    followed by a mild 0.996^(words-100) decay to suppress extreme verbosity.
    Multi-block tag exploits are penalized linearly by 0.5 per extra block.
    Zeroes out if the math answer is incorrect.
    """
    rewards = []

    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)

        if not is_correct:
            rewards.append(0.0)
            continue

        # 1. Extract thinking monologue content
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        num_start = comp_text.count("<think>")

        monologue_content = ""
        if think_blocks:
            monologue_content = think_blocks[0]
        elif num_start > 0:
            # Fallback for unclosed tag
            monologue_content = comp_text.split("<think>")[-1]

        # 2. Count actual words inside the monologue
        words = len(monologue_content.split())

        # 3. Apply 100-word grace window + mild 0.996 decay
        if words <= 100:
            conciseness_decay = 1.0
        else:
            conciseness_decay = 0.996 ** (words - 100)

        # 4. Multi-block penalty (arbitrage prevention)
        block_penalty = 0.0
        if num_start > 1:
            block_penalty = 0.5 * (num_start - 1)

        decayed_reward = max(0.0, conciseness_decay - block_penalty)
        rewards.append(decayed_reward)

    return rewards


# ── Tag-Spam Electrified Fence ──────────────────────────────────────────
_TAG_WHITELIST = frozenset(["think"])


def tag_spam_penalty_fn(prompts, completions, **kwargs) -> list[float]:
    """Aggressive penalty for ANY tag that is not <think> or </think>.
    Catches HTML tags, digit tags (<1>, <60>), step tags (<step1>),
    number-word tags (<one>), and arbitrary invented tags.
    Returns -0.3 per bad tag occurrence, capped at -1.5."""
    penalties = []
    # Pattern matches standard and custom XML tags (including hyphens, dots, colons, and digits)
    tag_pattern = re.compile(r"</?([a-zA-Z0-9_\-.:]+)>")

    for comp in completions:
        comp_text = get_completion_text(comp)
        bad_tag_count = 0
        bad_tag_types = set()

        for match in tag_pattern.finditer(comp_text):
            tag_name = match.group(1).lower()
            if tag_name not in _TAG_WHITELIST:
                bad_tag_types.add(tag_name)
                bad_tag_count += 1

        if bad_tag_count > 0:
            penalties.append(max(-1.5, -0.3 * bad_tag_count))
        else:
            penalties.append(0.0)

    return penalties
