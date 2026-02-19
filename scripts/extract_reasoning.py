#!/usr/bin/env python3
"""Extract reasoning from a Claude Code session transcript for a specific tool_use.

Self-contained helper called by capture-edits.sh hook in background.
Zero external dependencies - only uses stdlib.

Usage (from hook):
    python3 /path/to/extract_reasoning.py \
        --transcript-path /path/to/transcript.jsonl \
        --tool-use-id toolu_01ABC... \
        --timestamp 2026-02-06T10:00:00+01:00 \
        --tool Edit \
        --project my-project \
        --file-modified /path/to/file.py \
        --description "Edit: old -> new" \
        --output-dir /path/to/datalake/01-raw/code-changes/enriched

Exits silently on any error (must never block the hook).
"""

import argparse
import json
import sys
from pathlib import Path

MAX_WALK_DEPTH = 20


def load_transcript(transcript_path: str) -> tuple[dict, list]:
    """Load transcript, return (by_uuid dict, messages list)."""
    path = Path(transcript_path)
    if not path.exists():
        return {}, []

    messages = []
    by_uuid = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if obj.get("type") not in ("user", "assistant"):
                continue

            messages.append(obj)
            uuid = obj.get("uuid")
            if uuid:
                by_uuid[uuid] = obj

    return by_uuid, messages


def find_tool_use_message(messages: list, tool_use_id: str) -> dict | None:
    """Find the assistant message containing a specific tool_use block."""
    for msg in messages:
        if msg.get("message", {}).get("role") != "assistant":
            continue
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == tool_use_id
            ):
                return msg
    return None


def walk_up_for_context(by_uuid: dict, start_uuid: str) -> dict:
    """Walk up parentUuid chain collecting thinking and text context."""
    thinking_parts = []
    text_parts = []
    user_prompt = ""

    current = start_uuid
    for _ in range(MAX_WALK_DEPTH):
        msg = by_uuid.get(current)
        if not msg:
            break

        role = msg.get("message", {}).get("role", "")
        content = msg.get("message", {}).get("content", [])

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if isinstance(block, str) and role == "user":
                        user_prompt = block
                    continue

                btype = block.get("type", "")

                if btype == "thinking":
                    thinking_text = block.get("thinking", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)

                elif btype == "text" and role == "assistant":
                    text_val = block.get("text", "")
                    if text_val:
                        text_parts.append(text_val)

                elif btype == "text" and role == "user":
                    text_val = block.get("text", "")
                    if text_val:
                        user_prompt = text_val

        # Stop at original user message (not a tool_result)
        if role == "user":
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in (content if isinstance(content, list) else [])
            )
            if not has_tool_result:
                if isinstance(content, str):
                    user_prompt = content
                elif isinstance(content, list):
                    for b in content:
                        if isinstance(b, str):
                            user_prompt = b
                            break
                        if isinstance(b, dict) and b.get("type") == "text":
                            user_prompt = b.get("text", "")
                            break
                break

        current = msg.get("parentUuid")
        if not current:
            break

    thinking_parts.reverse()
    text_parts.reverse()

    return {
        "thinking": "\n\n".join(thinking_parts),
        "assistant_text": "\n\n".join(text_parts),
        "user_prompt": user_prompt,
    }


def determine_confidence(context: dict) -> str:
    """Determine enrichment confidence level."""
    thinking = context.get("thinking", "")
    text = context.get("assistant_text", "")

    if thinking and len(thinking) > 100:
        return "high"
    if text and len(text) > 50:
        return "medium"
    return "low"


def main():
    parser = argparse.ArgumentParser(description="Extract reasoning from transcript")
    parser.add_argument("--transcript-path", required=True)
    parser.add_argument("--tool-use-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--file-modified", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Load transcript
    by_uuid, messages = load_transcript(args.transcript_path)
    if not by_uuid:
        sys.exit(0)  # No transcript, silent exit

    # Find the tool_use message
    tool_msg = find_tool_use_message(messages, args.tool_use_id)
    if not tool_msg:
        sys.exit(0)  # Tool not found, silent exit

    # Walk up for context
    parent_uuid = tool_msg.get("parentUuid", "")
    if not parent_uuid:
        sys.exit(0)

    context = walk_up_for_context(by_uuid, parent_uuid)

    # Determine confidence
    confidence = determine_confidence(context)

    # Skip low confidence with no useful content
    if confidence == "low":
        sys.exit(0)

    # Build enriched record
    enriched = {
        "timestamp": args.timestamp,
        "type": "enriched_capture",
        "tool": args.tool,
        "project": args.project,
        "file_modified": args.file_modified,
        "description": args.description,
        "reasoning": context["thinking"],
        "assistant_context": context["assistant_text"],
        "user_prompt": context["user_prompt"],
        "enrichment_confidence": confidence,
        "tags": ["auto-captured", "enriched", "inline"],
    }

    # Write to enriched directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = args.timestamp[:10] if len(args.timestamp) >= 10 else "unknown"
    output_file = output_dir / f"{date_str}_enriched.jsonl"

    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(enriched, ensure_ascii=False, fp=f)
        f.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never fail - this runs in background from a hook
        sys.exit(0)
