#!/usr/bin/env python3
"""
Convert ShareGPT format to Qwen3-VL chat format.

Input format:
{"conversations": [{"from": "system/human/gpt", "value": "..."}]}

Output format:
{"messages": [{"role": "system/user/assistant", "content": "..."}]}
"""

import json
import argparse
from pathlib import Path


def convert_role(role: str) -> str:
    """Convert ShareGPT role names to Qwen format."""
    mapping = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
    }
    return mapping.get(role, role)


def convert_conversation(item: dict) -> dict:
    """Convert a single conversation from ShareGPT to Qwen format."""
    messages = []
    for turn in item.get("conversations", []):
        messages.append({
            "role": convert_role(turn["from"]),
            "content": turn["value"]
        })
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT to Qwen format")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="asuka_training_data.jsonl",
        help="Input JSONL file in ShareGPT format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file in Qwen format (default: input_qwen.jsonl)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_qwen")

    converted_count = 0
    skipped_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                converted = convert_conversation(item)

                # Validate: must have at least user + assistant
                roles = [m["role"] for m in converted["messages"]]
                if "user" not in roles or "assistant" not in roles:
                    print(f"Warning: Line {line_num} missing user or assistant, skipping")
                    skipped_count += 1
                    continue

                outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} has invalid JSON: {e}")
                skipped_count += 1
            except KeyError as e:
                print(f"Warning: Line {line_num} missing key {e}, skipping")
                skipped_count += 1

    print(f"\nConversion complete!")
    print(f"  Converted: {converted_count} conversations")
    print(f"  Skipped:   {skipped_count} conversations")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    main()
