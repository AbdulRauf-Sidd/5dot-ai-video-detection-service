"""
Re-run a user's uploaded videos with the original and the current (faster)
inference settings and compare the results, per chunk and per video.

Run inside the worker container on the GPU instance (needs DB env vars, the
model weights under /model-cache/{SERVICE_NAME}/ and S3 access):

    python compare_settings.py <email> [--limit N] [--list-only] [--csv out.csv]

    python compare_settings.py user@example.com --list-only     # check which videos are found
    python compare_settings.py user@example.com --limit 20

Settings compared:
    old  -- the original pipeline: RAFT_BATCH_SIZE=1, FLOW_PAIR_STRIDE=1
    new  -- --new-batch / --new-stride (default 8 / 2, the current production defaults)

The overall video score is the mean of the chunk probabilities, the same as
worker.py. The "stored" column is the score currently in detection_requests;
since settings changed over time it may match either run.

The DB connection is read-only: nothing is written back. Videos are downloaded
to a private temp dir, not the shared jobs dir live workers use.

detection_requests has no email column, so the script finds the link to the
users table from the DB's foreign keys and prints the join it uses. If that
fails, pass it explicitly with --user-table / --user-fk / --user-pk / --email-column.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import tempfile
import time

import psycopg2
import psycopg2.extras
from psycopg2 import sql

import db
from config.project_config import DEVICE, THRESHOLD
from helpers.video_helper import chunk_bounds, split_video_into_chunks
from ml_models.video import load_models
from repositories.validation_tool.validate import validate_video
from shared_storage import _download_source

OLD = {"batch": 1, "stride": 1}


# --------------------------------------------------------------------------
# Finding the user's videos
# --------------------------------------------------------------------------

def _columns(cur, table):
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s",
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def _email_column(cols):
    if "email" in cols:
        return "email"
    matches = [c for c in cols if "email" in c.lower()]
    return matches[0] if len(matches) == 1 else None


def discover_user_link(cur):
    """Return (user_table, fk_column_on_detection_requests, user_pk, email_column)."""
    # 1. Foreign keys from detection_requests to a table that has an email column
    cur.execute(
        """
        SELECT kcu.column_name, ccu.table_name, ccu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = current_schema()
          AND kcu.table_name = 'detection_requests'
        """
    )
    for fk_col, ref_table, ref_col in cur.fetchall():
        email_col = _email_column(_columns(cur, ref_table))
        if email_col:
            return ref_table, fk_col, ref_col, email_col

    # 2. No usable FK: guess from common column/table names
    dr_cols = _columns(cur, "detection_requests")
    fk_candidates = [c for c in ("user_id", "owner_id", "created_by", "uploaded_by", "account_id")
                     if c in dr_cols]
    for table in ("users", "user", "accounts", "auth_user"):
        cols = _columns(cur, table)
        email_col = _email_column(cols)
        if email_col and "id" in cols and fk_candidates:
            return table, fk_candidates[0], "id", email_col

    sys.exit(
        "Could not work out how detection_requests links to a user email.\n"
        f"detection_requests columns: {', '.join(dr_cols)}\n"
        "Pass it explicitly, e.g. --user-table users --user-fk user_id --user-pk id --email-column email"
    )


def fetch_videos(conn, email, args):
    with conn.cursor() as cur:
        if args.user_table:
            link = (args.user_table, args.user_fk or "user_id", args.user_pk, args.email_column)
        else:
            link = discover_user_link(cur)
    user_table, fk, pk, email_col = link
    print(f"[*] Joining detection_requests.{fk} -> {user_table}.{pk}, matching {user_table}.{email_col}")

    query = sql.SQL(
        """
        SELECT dr.id, dr.filename, dr.file_key, dr.url_source, dr.duration,
               dr.overall_ai_video_score, dr.ai_video_status, dr.created_at
        FROM detection_requests dr
        JOIN {ut} u ON dr.{fk} = u.{pk}
        WHERE lower(u.{ec}) = lower(%s)
          AND (dr.file_key IS NOT NULL OR dr.url_source IS NOT NULL)
        ORDER BY dr.created_at DESC
        LIMIT %s
        """
    ).format(ut=sql.Identifier(user_table), fk=sql.Identifier(fk),
             pk=sql.Identifier(pk), ec=sql.Identifier(email_col))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (email, args.limit))
        return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def infer(chunk, raft, fused, cfg, verbose):
    out = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    t0 = time.perf_counter()
    try:
        with out:
            r = validate_video(chunk, raft, fused, DEVICE, THRESHOLD,
                               raft_batch_size=cfg["batch"], flow_pair_stride=cfg["stride"])
        return r["probability"], time.perf_counter() - t0, None
    except Exception as e:  # e.g. CUDA OOM; the worker would score this chunk 0.0
        return None, time.perf_counter() - t0, str(e)


def verdict(p):
    return "-" if p is None else ("Fake" if p >= THRESHOLD else "Real")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(p, spec=".4f"):
    return "  n/a " if p is None else format(p, spec)


def compare_video(job, raft, fused, new_cfg, verbose, warmup=False):
    with tempfile.TemporaryDirectory(prefix=f"cmp_{job['id']}_") as work:
        ext = os.path.splitext(job.get("file_key") or "")[1] or ".mp4"
        source = os.path.join(work, "source" + ext)
        with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
            _download_source(job, source)
            chunks = split_video_into_chunks(source, os.path.join(work, "chunks"), 5)
        if warmup and chunks:
            # CUDA init / cuDNN autotuning would otherwise inflate the first timings
            infer(chunks[0], raft, fused, OLD, verbose)
            infer(chunks[0], raft, fused, new_cfg, verbose)
        rows = []
        for ch in chunks:
            old_p, old_t, old_err = infer(ch, raft, fused, OLD, verbose)
            new_p, new_t, new_err = infer(ch, raft, fused, new_cfg, verbose)
            start, end = chunk_bounds(ch)
            rows.append({
                "chunk": os.path.basename(ch), "start": start, "end": end,
                "old": old_p, "new": new_p, "old_t": old_t, "new_t": new_t,
                "error": old_err or new_err,
            })
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_video(i, n, job, rows):
    print()
    print(f"[{i}/{n}] {job['filename'] or job['id']}  (job {job['id']}, {len(rows)} chunks, "
          f"stored score {fmt(job['overall_ai_video_score'])}, status {job['ai_video_status']})")
    print(f"    {'chunk':<26}{'old':>8}{'new':>8}{'Δ':>9}  verdict")
    for r in rows:
        if r["error"]:
            print(f"    {r['chunk']:<26}{'ERROR':>8}  {r['error'][:80]}")
            continue
        d = r["new"] - r["old"]
        flip = "  FLIPPED" if verdict(r["old"]) != verdict(r["new"]) else ""
        print(f"    {r['chunk']:<26}{r['old']:>8.4f}{r['new']:>8.4f}{d:>+9.4f}  "
              f"{verdict(r['old'])} -> {verdict(r['new'])}{flip}")
    ok = [r for r in rows if not r["error"]]
    old_s, new_s = mean(r["old"] for r in ok), mean(r["new"] for r in ok)
    t_old, t_new = sum(r["old_t"] for r in rows), sum(r["new_t"] for r in rows)
    flip = "  FLIPPED" if verdict(old_s) != verdict(new_s) else ""
    d = "" if old_s is None or new_s is None else f"Δ {new_s - old_s:+.4f}"
    print(f"    {'overall (mean)':<26}{fmt(old_s):>8}{fmt(new_s):>8}{d:>11}  "
          f"{verdict(old_s)} -> {verdict(new_s)}{flip}   time {t_old:.1f}s -> {t_new:.1f}s")
    return old_s, new_s, t_old, t_new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("email")
    ap.add_argument("--limit", type=int, default=1000, help="max videos, newest first")
    ap.add_argument("--list-only", action="store_true", help="list the videos found and exit")
    ap.add_argument("--new-batch", type=int, default=8)
    ap.add_argument("--new-stride", type=int, default=2)
    ap.add_argument("--csv", default=f"settings_comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    ap.add_argument("--verbose", action="store_true", help="show inference/download logs")
    ap.add_argument("--user-table")
    ap.add_argument("--user-fk")
    ap.add_argument("--user-pk", default="id")
    ap.add_argument("--email-column", default="email")
    args = ap.parse_args()
    new_cfg = {"batch": args.new_batch, "stride": args.new_stride}

    conn = db.connect()
    conn.set_session(readonly=True, autocommit=True)
    try:
        jobs = fetch_videos(conn, args.email, args)
    finally:
        conn.close()

    print(f"[*] {len(jobs)} videos for {args.email}")
    for j in jobs:
        print(f"    {j['created_at']}  {j['id']}  {j['filename'] or j['file_key'] or j['url_source']}  "
              f"(stored {fmt(j['overall_ai_video_score'])}, {j['ai_video_status']})")
    if args.list_only or not jobs:
        return

    print(f"[*] old = batch {OLD['batch']}, stride {OLD['stride']} | "
          f"new = batch {new_cfg['batch']}, stride {new_cfg['stride']} | threshold {THRESHOLD}")
    raft, fused, _ = load_models()

    summary, all_rows = [], []
    warmed = False
    for i, job in enumerate(jobs, 1):
        try:
            rows = compare_video(job, raft, fused, new_cfg, args.verbose, warmup=not warmed)
            warmed = warmed or bool(rows)
        except Exception as e:
            print(f"\n[{i}/{len(jobs)}] {job['id']}: SKIPPED ({e})")
            continue
        old_s, new_s, t_old, t_new = print_video(i, len(jobs), job, rows)
        summary.append((job, rows, old_s, new_s, t_old, t_new))
        for r in rows:
            all_rows.append({"job_id": job["id"], "filename": job["filename"], **r})

    if not summary:
        print("\nNo videos could be processed.")
        return

    # --- Summary across videos ---
    print()
    print("=" * 100)
    print(f"{'video':<32}{'chunks':>7}{'old':>8}{'new':>8}{'Δ':>9}{'stored':>8}  "
          f"{'old -> new verdict':<20}{'speedup':>8}")
    video_flips, chunk_flips, n_chunks, errors = 0, 0, 0, 0
    video_d, chunk_d, t_old_all, t_new_all = [], [], 0.0, 0.0
    for job, rows, old_s, new_s, t_old, t_new in summary:
        name = (job["filename"] or str(job["id"]))[:30]
        flipped = verdict(old_s) != verdict(new_s)
        video_flips += flipped
        if old_s is not None and new_s is not None:
            video_d.append(abs(new_s - old_s))
        for r in rows:
            if r["error"]:
                errors += 1
                continue
            n_chunks += 1
            chunk_d.append(abs(r["new"] - r["old"]))
            chunk_flips += verdict(r["old"]) != verdict(r["new"])
        t_old_all += t_old
        t_new_all += t_new
        d = f"{new_s - old_s:+.4f}" if old_s is not None and new_s is not None else "n/a"
        v = f"{verdict(old_s)} -> {verdict(new_s)}" + (" *" if flipped else "")
        print(f"{name:<32}{len(rows):>7}{fmt(old_s):>8}{fmt(new_s):>8}{d:>9}"
              f"{fmt(job['overall_ai_video_score']):>8}  {v:<20}{(t_old / t_new if t_new else 0):>7.2f}x")

    print("=" * 100)
    print(f"videos: {len(summary)} | video verdict flips: {video_flips} | "
          f"chunk verdict flips: {chunk_flips}/{n_chunks} | chunk errors: {errors}")
    if video_d:
        print(f"overall score |Δ|: mean {sum(video_d) / len(video_d):.4f}, max {max(video_d):.4f}")
    if chunk_d:
        print(f"chunk score   |Δ|: mean {sum(chunk_d) / len(chunk_d):.4f}, max {max(chunk_d):.4f}")
    print(f"inference time: old {t_old_all:.1f}s, new {t_new_all:.1f}s "
          f"({t_old_all / t_new_all if t_new_all else 0:.2f}x)")
    print("* = video verdict differs. Timings include the worker container's own load, if any.")

    with open(args.csv, "w", newline="") as f:
        fields = ["job_id", "filename", "chunk", "start", "end", "old", "new", "delta",
                  "old_verdict", "new_verdict", "old_t", "new_t", "error"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        r6 = lambda x: round(x, 6) if isinstance(x, float) else x
        w.writerow = (lambda orig: lambda row: orig({k: r6(v) for k, v in row.items()}))(w.writerow)
        for r in all_rows:
            ok = not r["error"]
            w.writerow({**{k: r.get(k) for k in fields},
                        "delta": (r["new"] - r["old"]) if ok else None,
                        "old_verdict": verdict(r["old"]), "new_verdict": verdict(r["new"])})
        for job, rows, old_s, new_s, *_ in summary:
            w.writerow({"job_id": job["id"], "filename": job["filename"], "chunk": "OVERALL",
                        "old": old_s, "new": new_s,
                        "delta": (new_s - old_s) if old_s is not None and new_s is not None else None,
                        "old_verdict": verdict(old_s), "new_verdict": verdict(new_s)})
    print(f"[*] Per-chunk results written to {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
