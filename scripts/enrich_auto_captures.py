#!/usr/bin/env python3
"""Enrich auto-captures with reasoning from session transcripts.

Reads auto-capture records and extracts thinking/text context from
Claude Code session transcripts by walking the parentUuid chain.

Output goes to a separate 'enriched/' subdirectory - NEVER modifies originals.

Usage:
    python3 scripts/enrich_auto_captures.py --dry-run
    python3 scripts/enrich_auto_captures.py --project my-project
    python3 scripts/enrich_auto_captures.py  # all projects
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge.extraction.pipeline import _parse_multiline_json


DATALAKE = Path(os.environ.get("FABRIK_DATALAKE_PATH", Path(__file__).resolve().parent.parent / "data"))
TRANSCRIPTS_BASE = Path.home() / ".claude" / "projects"
MAX_WALK_DEPTH = 20
TIMESTAMP_FUZZY_WINDOW = timedelta(seconds=120)


def load_transcript(transcript_path: str | Path) -> dict:
    """Load a transcript file and return uuid-indexed messages.

    Returns:
        Dict with keys:
        - 'by_uuid': {uuid: message_dict}
        - 'messages': [message_dict, ...]  (ordered)
    """
    path = Path(transcript_path)
    if not path.exists():
        return {"by_uuid": {}, "messages": []}

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

    return {"by_uuid": by_uuid, "messages": messages}


def find_tool_use_message(transcript: dict, tool_use_id: str) -> dict | None:
    """Find the assistant message containing a specific tool_use block."""
    for msg in transcript["messages"]:
        if msg.get("message", {}).get("role") != "assistant":
            continue
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("id") == tool_use_id:
                    return msg
    return None


def walk_up_for_context(
    transcript: dict,
    start_uuid: str,
    max_depth: int = MAX_WALK_DEPTH,
) -> dict:
    """Walk up the parentUuid chain collecting thinking and text context.

    Returns:
        {
            'thinking': str,        # concatenated thinking blocks
            'assistant_text': str,  # concatenated text blocks from assistant
            'user_prompt': str,     # the original user message
        }
    """
    by_uuid = transcript["by_uuid"]
    thinking_parts = []
    text_parts = []
    user_prompt = ""

    current = start_uuid
    for _ in range(max_depth):
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
                    if text_val and "tool_result" not in str(content[:1]):
                        user_prompt = text_val

        # If we hit the original user message (not a tool_result), stop
        if role == "user":
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in (content if isinstance(content, list) else [])
            )
            if not has_tool_result:
                # This is the original user prompt
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

    # Reverse since we walked bottom-up
    thinking_parts.reverse()
    text_parts.reverse()

    return {
        "thinking": "\n\n".join(thinking_parts),
        "assistant_text": "\n\n".join(text_parts),
        "user_prompt": user_prompt,
    }


def fuzzy_match_in_transcript(
    transcript: dict,
    timestamp_str: str,
    file_modified: str,
) -> dict | None:
    """Find an Edit/Write tool_use by fuzzy timestamp + file match.

    Used for records that don't have tool_use_id (recovered data).
    """
    try:
        # Parse timestamp - handle various formats
        ts = timestamp_str.replace("Z", "+00:00")
        if "T" in ts:
            target_dt = datetime.fromisoformat(ts)
        else:
            return None
    except (ValueError, TypeError):
        return None

    best_match = None
    best_delta = TIMESTAMP_FUZZY_WINDOW

    for msg in transcript["messages"]:
        if msg.get("message", {}).get("role") != "assistant":
            continue

        msg_ts_str = msg.get("timestamp", "")
        if not msg_ts_str:
            continue

        try:
            msg_ts = datetime.fromisoformat(msg_ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        delta = abs(msg_ts - target_dt)
        if delta > TIMESTAMP_FUZZY_WINDOW:
            continue

        # Check if this message has an Edit/Write tool_use for the same file
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in ("Edit", "Write"):
                continue

            tool_file = block.get("input", {}).get("file_path", "")
            if file_modified and tool_file and (
                tool_file == file_modified
                or Path(tool_file).name == Path(file_modified).name
            ):
                if delta < best_delta:
                    best_delta = delta
                    best_match = msg

    return best_match


def find_transcript_for_project(project: str) -> list[Path]:
    """Find all transcript files that might contain edits for a project."""
    transcripts = []

    # Claude Code stores transcripts by project path
    # e.g., ~/.claude/projects/-home-user-projects-myproject/
    for project_dir in TRANSCRIPTS_BASE.iterdir():
        if not project_dir.is_dir():
            continue
        dir_name = project_dir.name.lower()
        if project.lower().replace("-", "") in dir_name.replace("-", ""):
            for jsonl in project_dir.glob("*.jsonl"):
                transcripts.append(jsonl)

    return sorted(transcripts)


def determine_confidence(context: dict) -> str:
    """Determine enrichment confidence level."""
    thinking = context.get("thinking", "")
    text = context.get("assistant_text", "")

    if thinking and len(thinking) > 100:
        return "high"
    if text and len(text) > 50:
        return "medium"
    return "low"


def enrich_record(record: dict, transcript_cache: dict) -> dict | None:
    """Enrich a single auto-capture record with reasoning from transcript.

    Returns enriched record or None if no enrichment possible.
    """
    transcript_path = record.get("transcript_path", "")
    tool_use_id = record.get("tool_use_id", "")
    project = record.get("project", "")
    timestamp = record.get("timestamp", "")
    file_modified = record.get("file_modified", "")

    context = None

    # Strategy 1: Direct lookup via transcript_path + tool_use_id
    if transcript_path and tool_use_id:
        if transcript_path not in transcript_cache:
            transcript_cache[transcript_path] = load_transcript(transcript_path)
        transcript = transcript_cache[transcript_path]

        tool_msg = find_tool_use_message(transcript, tool_use_id)
        if tool_msg:
            parent_uuid = tool_msg.get("parentUuid", "")
            if parent_uuid:
                context = walk_up_for_context(transcript, parent_uuid)

    # Strategy 2: Fuzzy match by timestamp + file in project transcripts
    if not context or (not context.get("thinking") and not context.get("assistant_text")):
        if project and project != "unknown":
            project_transcripts = find_transcript_for_project(project)
            for tp in project_transcripts:
                tp_str = str(tp)
                if tp_str not in transcript_cache:
                    transcript_cache[tp_str] = load_transcript(tp)
                transcript = transcript_cache[tp_str]

                matched_msg = fuzzy_match_in_transcript(
                    transcript, timestamp, file_modified,
                )
                if matched_msg:
                    parent_uuid = matched_msg.get("parentUuid", "")
                    if parent_uuid:
                        context = walk_up_for_context(transcript, parent_uuid)
                        if context.get("thinking") or context.get("assistant_text"):
                            break

    if not context:
        return None

    confidence = determine_confidence(context)

    # Skip "low" confidence with no useful content
    if confidence == "low" and not context.get("user_prompt"):
        return None

    enriched = {
        "timestamp": timestamp,
        "type": "enriched_capture",
        "tool": record.get("tool", ""),
        "project": project,
        "file_modified": file_modified,
        "description": record.get("description", ""),
        "reasoning": context.get("thinking", ""),
        "assistant_context": context.get("assistant_text", ""),
        "user_prompt": context.get("user_prompt", ""),
        "enrichment_confidence": confidence,
        "tags": ["auto-captured", "enriched"],
    }

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich auto-captures with reasoning")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--project", type=str, help="Filter by project name")
    parser.add_argument(
        "--datalake",
        type=str,
        default=str(DATALAKE),
        help="Datalake path",
    )
    args = parser.parse_args()

    datalake = Path(args.datalake)
    auto_dir = datalake / "01-raw" / "code-changes"

    if not auto_dir.exists():
        print(f"Error: auto-captures directory not found: {auto_dir}")
        sys.exit(1)

    # Find all auto-capture files
    cap_files = sorted(auto_dir.glob("*auto-captures*.jsonl"))
    if not cap_files:
        print("No auto-capture files found.")
        return

    print(f"Found {len(cap_files)} auto-capture files")

    # Prepare enriched output directory
    enriched_dir = auto_dir / "enriched"
    if not args.dry_run:
        enriched_dir.mkdir(exist_ok=True)

    transcript_cache: dict = {}
    stats = defaultdict(int)

    for cap_file in cap_files:
        records = _parse_multiline_json(cap_file)
        enriched_records = []

        for record in records:
            if record.get("type") != "auto_capture":
                continue

            project = record.get("project", "")
            if args.project and project != args.project:
                continue

            stats["total"] += 1

            enriched = enrich_record(record, transcript_cache)
            if enriched:
                confidence = enriched["enrichment_confidence"]
                stats[f"enriched_{confidence}"] += 1
                stats["enriched_total"] += 1
                enriched_records.append(enriched)
            else:
                stats["no_match"] += 1

        if enriched_records and not args.dry_run:
            # Write enriched records grouped by date
            by_date: dict[str, list] = defaultdict(list)
            for rec in enriched_records:
                ts = rec.get("timestamp", "")
                date_str = ts[:10] if len(ts) >= 10 else "unknown"
                by_date[date_str].append(rec)

            for date_str, recs in by_date.items():
                out_file = enriched_dir / f"{date_str}_enriched.jsonl"
                with open(out_file, "a", encoding="utf-8") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if enriched_records:
            print(f"  {cap_file.name}: {len(enriched_records)}/{len(records)} enriched")

    print(f"\n--- Summary ---")
    print(f"Total records scanned: {stats['total']}")
    print(f"Enriched (total):      {stats['enriched_total']}")
    print(f"  High confidence:     {stats['enriched_high']}")
    print(f"  Medium confidence:   {stats['enriched_medium']}")
    print(f"  Low confidence:      {stats['enriched_low']}")
    print(f"No match:              {stats['no_match']}")

    if args.dry_run:
        print("\n(dry-run mode - nothing written)")
    else:
        print(f"\nOutput: {enriched_dir}/")


if __name__ == "__main__":
    main()
