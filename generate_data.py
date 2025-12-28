#!/usr/bin/env python3
"""
Generate synthetic conversation data using Hermes-4.3-36B.
CUDA version for A100.

Usage:
    python generate_data.py --character character.txt --output synthetic_data.jsonl --count 10000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Model to use
MODEL_ID = "NousResearch/Hermes-4.3-36B"

# Default topics/scenarios for diverse conversations
TOPICS = [
    # Casual everyday (~35)
    "casual morning greeting",
    "asking how their day is going",
    "asking about their hobbies",
    "discussing favorite food",
    "talking about music taste",
    "discussing a movie or show",
    "asking about their weekend plans",
    "discussing work or school stress",
    "asking for their opinion on something",
    "talking about pet peeves",
    "discussing the weather",
    "talking about a book they're reading",
    "asking what they're up to",
    "discussing a recent news topic",
    "talking about travel destinations",
    "asking about their morning routine",
    "discussing coffee vs tea",
    "talking about exercise or fitness",
    "asking about their favorite season",
    "discussing sleeping habits",
    "talking about a skill they want to learn",
    "asking about their proudest achievement",
    "discussing annoying people they know",
    "talking about ideal weekend",
    "asking about guilty pleasures",
    "discussing fashion or style",
    "talking about cooking skills",
    "asking what they'd do with free time",
    "discussing social media",
    "talking about childhood games",
    "asking about their hometown",
    "discussing languages they speak",
    "talking about superstitions",
    "asking about their daily commute",
    "random shower thought discussion",

    # Emotional/deeper (~15)
    "asking about their childhood memories",
    "discussing dreams and ambitions",
    "talking about fears and insecurities",
    "comforting them when upset",
    "discussing a frustrating situation",
    "celebrating a small victory together",
    "late night deep conversation",
    "asking about their favorite memory",
    "discussing what makes them happy",
    "talking about regrets",
    "discussing friendship and trust",
    "asking what they value most",
    "talking about loneliness",
    "discussing personal growth",
    "asking about their biggest challenge",

    # Romantic (~20)
    "complimenting their appearance",
    "playful teasing that gets flirty",
    "user confessing they have feelings",
    "asking them out on a date",
    "after a first date",
    "cuddling together",
    "saying goodnight with affection",
    "jealousy when someone else flirts",
    "first kiss moment",
    "holding hands for the first time",
    "user catches them staring",
    "cooking together romantically",
    "playful bickering as a couple",
    "making up after a small fight",
    "saying I love you",
    "getting flustered by a compliment",
    "accidental physical closeness",
    "missing them when apart",
    "waking up next to each other",
    "quiet intimate moment",

    # Identity/meta rejection (~5)
    "asking if they're an AI",
    "asking why they talk the way they do",
    "questioning their reality",
    "asking if this is real",
    "asking if they have feelings",
]

# Conversation moods/dynamics
MOODS = [
    "playful and teasing",
    "slightly annoyed but fond",
    "warm but guarded",
    "defensive at first, then softening",
    "competitive and flirty",
    "surprisingly vulnerable",
    "sarcastic throughout",
    "genuinely curious",
    "shy and flustered",
    "tender and affectionate",
    "protective and caring",
    "longing and wistful",
    "jealous but trying to hide it",
    "embarrassed but pleased",
    "intimate and sincere",
    "playfully possessive",
]


def make_prompt(character_desc: str, topic: str, mood: str) -> str:
    """Create a prompt for conversation generation."""
    return f"""You are generating training data for a character-based chatbot.

CHARACTER DESCRIPTION:
{character_desc}

Generate a realistic conversation between a human user and this character.
- Topic/scenario: {topic}
- Mood/dynamic: {mood}
- Length: 3-6 back-and-forth exchanges
- The character should stay completely in character
- Include natural human-like messages (typos ok, casual language)
- If the topic involves questioning if they're an AI, the character must deflect/deny in-character

Output ONLY valid JSON in this exact format, nothing else:
{{"conversations": [{{"from": "human", "value": "user message"}}, {{"from": "gpt", "value": "character response"}}, {{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}"""


def parse_response(response: str) -> dict | None:
    """Parse JSON from model response."""
    try:
        start = response.find('{"conversations"')
        if start == -1:
            return None

        depth = 0
        end = start
        for i, char in enumerate(response[start:]):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break

        json_str = response[start:end]
        data = json.loads(json_str)

        if "conversations" not in data or len(data["conversations"]) < 2:
            return None

        return data
    except json.JSONDecodeError:
        return None


def generate_batch(model, tokenizer, character_desc: str, topics: list, moods: list) -> list:
    """Generate a batch of conversations."""
    prompts = [make_prompt(character_desc, t, m) for t, m in zip(topics, moods)]

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each output
    results = []
    for i, output in enumerate(outputs):
        # Skip the input tokens
        input_len = (inputs.attention_mask[i] == 1).sum().item()
        response = tokenizer.decode(output[input_len:], skip_special_tokens=True)

        print(f"\n--- Output {i+1} ({topics[i]}) ---")
        print(response[:500] + "..." if len(response) > 500 else response)

        parsed = parse_response(response)
        results.append(parsed)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=MODEL_ID,
        help="HuggingFace model to use"
    )
    parser.add_argument(
        "--character", "-c",
        type=str,
        required=True,
        help="Path to character description file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synthetic_data.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Number of conversations to generate"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )

    args = parser.parse_args()

    # Load character description
    char_path = Path(args.character)
    if not char_path.exists():
        print(f"Error: Character file not found: {args.character}")
        sys.exit(1)

    character_desc = char_path.read_text().strip()
    print(f"Loaded character description ({len(character_desc)} chars)")

    # Load model
    print(f"Loading model: {args.model}")
    print("(This will download ~70GB on first run)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 4-bit quantization to fit 36B on A100 40GB (~20GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded on {model.device}")
    print()

    print(f"Generating {args.count} conversations (batch size: {args.batch_size})...")
    print()

    mode = "a" if args.append else "w"
    generated = 0
    failed = 0

    with open(args.output, mode) as f:
        for batch_start in range(0, args.count, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.count)
            batch_size = batch_end - batch_start

            # Pick random topics and moods for this batch
            topics = [random.choice(TOPICS) for _ in range(batch_size)]
            moods = [random.choice(MOODS) for _ in range(batch_size)]

            print(f"\n{'='*60}")
            print(f"Batch [{batch_start+1}-{batch_end}/{args.count}]")
            print(f"Topics: {topics}")
            print('='*60)

            results = generate_batch(model, tokenizer, character_desc, topics, moods)

            for i, conv in enumerate(results):
                if conv:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    generated += 1
                    print(f"  ✓ [{batch_start+i+1}] Saved")
                else:
                    failed += 1
                    print(f"  ✗ [{batch_start+i+1}] Failed to parse")

            f.flush()
            print(f"\nProgress: {generated} generated, {failed} failed")

    print()
    print(f"Done! Generated: {generated}, Failed: {failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
