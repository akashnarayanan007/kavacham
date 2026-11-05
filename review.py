"""Record a person's review of the model's verdicts, and rebuild verified_conflicts.json.

Usage, in a run folder:
    python review.py --not-real 12 --reason "Steady gap for 10 sec: travelling together"
    python review.py --real 70 --reason "Checked the clip: genuine close pass"
    python review.py --clear 12

Writes human_review.json and verified_conflicts.json. The model's own verdicts in
llama_raw_responses.json are never changed, so what it said stays on record.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import REVIEW_JSON, apply_reviews, build_verified, load_reviews

CONFLICTS_JSON = Path("potential_conflicts.json")
RAW_JSON = Path("llama_raw_responses.json")
VERIFIED_JSON = Path("verified_conflicts.json")


def main():
    parser = argparse.ArgumentParser(description="Review model verdicts on conflicts.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--real", type=int, metavar="ID", help="confirm a conflict is real")
    action.add_argument("--not-real", type=int, metavar="ID", help="mark a conflict as normal")
    action.add_argument("--clear", type=int, metavar="ID", help="remove a review")
    parser.add_argument("--reason", help="what the reviewer saw (required unless --clear)")
    parser.add_argument("--reviewer", default="person", help="who reviewed it")
    args = parser.parse_args()

    try:
        conflicts = json.loads(CONFLICTS_JSON.read_text(encoding="utf-8"))
        records = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"Run this in a run folder that verify.py has finished: {error}")

    conflict_id = args.real or args.not_real or args.clear
    record = next((r for r in records if r["conflict_id"] == conflict_id), None)
    if record is None or not record.get("verdict"):
        sys.exit(f"Conflict {conflict_id} has no model verdict to review.")

    reviews = load_reviews(".")
    if args.clear is not None:
        reviews.pop(conflict_id, None)
    else:
        if not args.reason:
            parser.error("--reason is required")
        reviews[conflict_id] = {
            "verdict": {"is_real": args.real is not None, "reason": args.reason},
            "reviewer": args.reviewer,
            "model_verdict": record["verdict"],
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    Path(REVIEW_JSON).write_text(json.dumps({str(k): v for k, v in sorted(reviews.items())},
                                            indent=2), encoding="utf-8")

    verified = build_verified(conflicts, apply_reviews(records, reviews))
    VERIFIED_JSON.write_text(json.dumps(verified, indent=2), encoding="utf-8")
    model_real = sum(1 for r in records if r.get("verdict") and r["verdict"]["is_real"])
    print(f"{len(reviews)} reviewed. Real near-misses: {len(verified)} after review "
          f"(the model said {model_real}).")
    for row in verified:
        print(f'  #{row["conflict_id"]} {row["label_1"]} / {row["label_2"]}, severity '
              f'{row["severity"]}, {row["risk_type"]}')


if __name__ == "__main__":
    main()
