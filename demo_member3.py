"""
demo_member3.py - End-to-end demo / integration test of Mode D (no GUI)
======================================================================
Runs real files from the dataset through hierarchical_verify() and then through
Member 2's command gate, for four scenarios:

  1. Genuine user (face + voice + fingerprint of session 3)  -> Level 3, commands run per role
  2. Total intruder (unseen face + unseen voice + unseen finger) -> stopped at Level 1
  3. Right face, WRONG voice                                   -> stopped at Level 2
  4. Right face + voice, WRONG fingerprint                     -> Level 2 only (no Level-3 permission)

In every scenario a voice command is attempted afterwards. The script exits with an
error if an unverified user is ever allowed to execute a command.

Prerequisite:  python run_member3_experiments.py       (creates models/hierarchy/*)
Usage:         python demo_member3.py [--user member1] [--command logs]

Note: execute_voice_command() appends to logs/audit_log.csv (audit trail of Bonus 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.command_system import execute_voice_command
from src.hierarchy import hierarchical_verify, load_bundle
from src.multimodal_dataset import build_rows, load_pools


def pick(pool, persons, sessions=None, raw_only=True, k=0):
    idx = np.flatnonzero(pool.mask(persons, sessions, raw_only))
    return str(pool.paths[idx[k % len(idx)]])


def command_audio(user, cmd, session="session3"):
    files = sorted((ROOT / "data" / "commands" / user / session / cmd).glob("*.wav"))
    return str(files[0]) if files else cmd  # fall back to the command name


def run_scenario(title, user, face_path, voice_path, fp_path, cmd, expect_level, expect_decision):
    print(f"\n--- {title} ---")
    result = hierarchical_verify(user, cv2.imread(face_path), voice_path, fp_path)
    sc = {k: (None if v is None else round(v, 3)) for k, v in result["scores"].items()}
    print(f"  claimed identity : {user}")
    print(f"  scores           : {sc}   thresholds: { {k: round(v, 3) for k, v in result['thresholds'].items()} }")
    print(f"  level reached    : {result['level_reached']}   decision: {'ACCESS GRANTED' if result['decision'] else 'ACCESS DENIED'}"
          f"   stopped at: {result['stopped_at']}")
    print(f"  time per stage   : { {k: round(v, 1) for k, v in result['timing_ms'].items()} } ms")
    cmd_res = execute_voice_command(command_audio(user, cmd), user_id=user,
                                    is_verified=result["decision"], level_reached=result["level_reached"])
    print(f"  command '{cmd}'  : {cmd_res['status']} - {cmd_res['message']}")

    ok = (result["level_reached"] == expect_level) and (result["decision"] == expect_decision)
    blocked_ok = result["decision"] or cmd_res["status"] == "ACCESS_DENIED"   # unverified => must be blocked
    print(f"  expectation      : level {expect_level}, decision {expect_decision}  -> {'OK' if ok else 'DIFFERENT'};"
          f" unverified-blocked check -> {'OK' if blocked_ok else 'FAIL'}")
    return ok, blocked_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="member1")
    ap.add_argument("--command", default="logs", help="command attempted after verification (open/lock/logs/status)")
    args = ap.parse_args()

    load_bundle(args.user)  # fail early with a clear message if not trained
    pools = load_pools()
    _, test, info = build_rows(args.user, pools)
    unseen = info["plan"]

    u = args.user
    face_ok = pick(pools.face, [u], ["session3"])
    voice_ok = pick(pools.voice, [u], ["session3"])
    fp_ok = pick(pools.fp, [u], ["session3"])
    face_bad = pick(pools.face, unseen["unseen_face"][:1])
    voice_bad = pick(pools.voice, unseen["unseen_voice"][:1])
    fp_bad = pick(pools.fp, unseen["unseen_fp"][:1])

    print("=" * 74 + f"\n SMART SECURITY GATE - hierarchical verification demo (claimed user: {u})\n" + "=" * 74)
    outcomes = [
        run_scenario("1. Genuine user", u, face_ok, voice_ok, fp_ok, args.command, 3, True),
        run_scenario("2. Intruder: unseen face + voice + fingerprint", u, face_bad, voice_bad, fp_bad, args.command, 0, False),
        run_scenario("3. Right face, wrong voice", u, face_ok, voice_bad, fp_ok, args.command, 1, False),
        run_scenario("4. Right face + voice, wrong fingerprint", u, face_ok, voice_ok, fp_bad, args.command, 2, True),
    ]
    print("\n" + "=" * 74)
    if not all(b for _, b in outcomes):
        print("FAIL: an unverified user was able to run a command!")
        sys.exit(1)
    matched = sum(a for a, _ in outcomes)
    print(f"Security check passed: unverified users were blocked in every scenario.\n"
          f"Scenarios matching the expected level: {matched}/{len(outcomes)} "
          f"(a miss here is a real FRR/FAR event of the trained models, not a code error).")


if __name__ == "__main__":
    main()
