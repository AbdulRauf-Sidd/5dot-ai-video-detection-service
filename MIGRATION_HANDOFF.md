# Worker migration handoff — lip_sync / scene_detection

`video_ai_service` was just converted from FastAPI+Celery+SQLAlchemy to a
standalone long-polling SQS worker script. This doc is for whoever (agent or
human) does the same conversion on the `lip_sync` and `scene_detection`
repos, so the same architectural decisions get applied consistently instead
of being re-derived (and possibly re-decided differently) from scratch.

Give the agent working on those repos this whole file as context, then let
it explore that repo's actual code before writing anything — don't assume
file names/structure match `video_ai_service` 1:1.

Reference implementation (read these before writing new code):
`video_ai_service/worker.py`, `db.py`, `shared_storage.py`, `webhook.py`,
`config/project_config.py`, `ml_models/video.py`, `Dockerfile`,
`requirements.txt`, `test.py`.

---

## 1. Original architecture spec (applies to all 3 services)

Each service is a standalone Python script running an infinite loop:

1. Long-poll its own SQS queue (each service has a dedicated queue).
2. On message: extract `job_id`, query Postgres for the job record (raw SQL
   via psycopg2 — connection opened once at startup, reused across the loop).
3. Download the source file into a shared local folder (`/tmp/shared_jobs`)
   using a lock-file pattern: if another service is already downloading this
   `job_id`, wait for a `.done` marker instead of downloading again. The
   folder is genuinely shared — all 3 services run as separate Docker
   containers on the same EC2 instance, mounting the same `/tmp`.
4. Run inference (chunking logic already implemented per service).
5. Write results to Postgres via raw SQL (`UPDATE`, not ORM).
6. POST a webhook to the core service with
   `{job_id, service_name, status, result}` — `https://core-service.placeholder/webhook`
   for now. If the webhook fails, log the error and retry up to 3 times with
   backoff, but still delete the SQS message either way (result is already
   durably saved in Postgres).
7. Delete the SQS message.

Also:
- Model weights are **not** downloaded — baked into the AMI at
  `/model-cache/{service_name}/`. Load from that path at startup, offline.
- Config via env vars: `SQS_QUEUE_URL`, `DB_HOST`, `DB_NAME`, `DB_USER`,
  `DB_PASSWORD`, `WEBHOOK_URL`, `SERVICE_NAME`, `IDLE_TIMEOUT_SECONDS`.

Cloud context (not built yet, informs design): core service runs on its own
EC2 instance (separate project); one shared EC2 instance runs all 3
inference services as separate Docker containers; each service has its own
Dockerfile + requirements.txt (different Python versions/deps); images are
built and pushed to ECR, the instance pulls at boot.

---

## 2. Decisions already made — apply identically unless the target repo's
   real facts contradict them

These came out of a back-and-forth with the user during the
`video_ai_service` build. They're cross-service architectural agreements,
not video-specific judgment calls, so don't re-litigate them — just confirm
the target repo doesn't have a conflicting existing convention.

- **Replace, don't keep alongside.** Old FastAPI/Celery/SQLAlchemy code gets
  deleted entirely, not left running side-by-side with the new worker.
- **Shared Postgres schema.** One `detection_requests` row per job, shared
  across all detection services, plus a shared `detection_chunks` table
  keyed on `(detection_request_id, chunk_index)`. Each service reads/writes
  only its own columns on both tables.
- **Chunk rows are pre-created, never inserted by the worker.** No unique
  constraint exists (and none should be added) on
  `(detection_request_id, chunk_index)` — something upstream of these 3
  workers (probably the core service, when it creates the job) already
  inserts one chunk row per `chunk_index` with `segment_start`/`segment_end`
  filled in. Each detection worker only ever does `UPDATE ... WHERE
  detection_request_id = %s AND chunk_index = %s` to fill in its own score
  column(s). **Verify this in the target repo/DB rather than assuming** —
  if `lip_sync`'s chunk boundaries could ever differ from `video_ai_service`'s
  for the same job, the pre-creation assumption breaks and needs revisiting.
- **No shared `result_data` JSONB.** Each service has its own dedicated
  top-level result column(s) on `detection_requests`
  (video's is `overall_ai_video_score`, a `Float` storing raw probability
  0.0–1.0, not 0–100). Don't read/write the shared `result_data` column.
- **Worker never touches the aggregate `status` column** on
  `detection_requests` — only its own `<service>_status` column. Assume the
  core service (or something else) computes the aggregate from all the
  per-service status columns.
- **Processing-failure policy** (distinct from webhook-failure policy):
  catch the exception, retry in-process up to 2 times (3 attempts total) for
  transient errors like CUDA OOM, then if still failing: mark
  `<service>_status = 'failed'`, write `error_message`, send webhook with
  `status=failed`, and **delete the SQS message regardless** — don't rely on
  SQS redrive/DLQ for inference failures.
- **`IDLE_TIMEOUT_SECONDS` is a heartbeat log only** — logs "idle, no
  messages in Ns" periodically, no effect on control flow (doesn't exit the
  process, doesn't trigger cleanup by itself).
- **Lock-file wait has a fixed timeout, then fails the job** — doesn't try
  to detect/reclaim a stale lock from a crashed downloader. Default
  `DOWNLOAD_WAIT_TIMEOUT_SECONDS = 600`, poll every 2s.
- **Shared file cleanup is DB-driven, done by the last finisher.** After a
  service finishes (success or fail) it checks Postgres: if every other
  *requested* detection type on that job (per the `detect_*` boolean flags)
  has already reached a terminal status (`complete`/`failed`), it deletes
  the whole `/tmp/shared_jobs/{job_id}/` folder. Otherwise it leaves the
  folder for the others. No separate periodic cleanup process exists.
- **DB connection**: opened once at startup (`psycopg2.connect(...)`,
  `autocommit = True`), reused for the life of the process. No reconnect
  logic was added — not asked for, keep it that way unless you hit a real
  need.
- **Column names are hardcoded per service, not derived from
  `SERVICE_NAME`.** The DB naming isn't consistent across services (video →
  `ai_video_*`, lip_sync → `lipsync_*`, scene_detection → `changes_*`) — see
  the open question below about this mismatch. Don't try to
  string-template column names from `SERVICE_NAME`.

---

## 3. Known column mapping (from `detection_requests` / `detection_chunks`)

```
detection_requests
├─ id, filename, file_key, url_source, file_size, duration, bitrate
├─ detect_ai_audio     / ai_audio_status   / overall_ai_audio_score   (not our 3 services — separate audio service)
├─ detect_ai_video     / ai_video_status   / overall_ai_video_score   (video_ai_service — DONE)
├─ detect_lipsync      / lipsync_status    / overall_lipsync_score    (lip_sync — TODO)
├─ detect_changes      / changes_status    (no overall_*_score column shown for this one — TODO confirm)
├─ status (aggregate — nobody but the core service touches this)
├─ thumbnail_key, result_data (JSONB, unused by any of the 3 workers), error_message (shared column — see open question)
└─ created_at, completed_at

detection_chunks
├─ id, detection_request_id, chunk_index, segment_start, segment_end
├─ ai_audio_score / ai_audio_start / ai_audio_end
├─ ai_video_score / ai_video_start / ai_video_end     (video_ai_service — DONE)
├─ lipsync_score  / lipsync_start  / lipsync_end       (lip_sync — TODO)
└─ changes_points (JSONB — scene_detection — TODO, structurally different from the score/start/end pattern)
```

---

## 4. Open questions — do NOT assume, ask the user before building

These weren't fully resolved even for `video_ai_service`, or are specific
enough to `lip_sync`/`scene_detection` that they need fresh answers:

1. **`SERVICE_NAME` value vs. DB column prefix mismatch.** The directory/
   service is called `lip_sync` but its DB columns are `lipsync_*` (no
   underscore). Similarly `scene_detection` vs. `changes_*`. Confirm the
   exact `SERVICE_NAME` string each container will be started with (used
   for `/model-cache/{SERVICE_NAME}/` and the webhook's `service_name`
   field) — it does **not** need to match the DB column prefix, but the
   agent must not conflate the two or try to derive one from the other.
2. **`scene_detection` has no `overall_changes_score` column.** Ask what its
   webhook `result` payload and any top-level "done" signal should actually
   contain — likely just the per-chunk `changes_points`, but confirm,
   since `changes_points` is a JSONB blob per chunk rather than a
   score/start/end triple, so the update logic will look meaningfully
   different from `video_ai_service`'s `update_chunk`.
3. **`error_message` is a single shared column** on `detection_requests`
   across all services — if two services fail on the same job, the second
   failure overwrites the first one's message. This was never explicitly
   decided for `video_ai_service` either (I made a pragmatic call to just
   write to it); flag it to the user for all 3 services at once rather than
   deciding per-repo.
4. **Existing scaffolding in the target repo** — don't assume `lip_sync`/
   `scene_detection` currently look like `video_ai_service` did (FastAPI +
   Celery + SQLAlchemy `Scan`/`User` models). Read the actual repo first.
   Replace-vs-keep-alongside was decided as "replace" for `video_ai_service`
   specifically — confirm it still holds here, since the user may have a
   different WIP state in these repos.
5. **Model weight files / loading pattern.** `video_ai_service` loads
   RAFT + XCLIP + a fused head. `lip_sync` and `scene_detection` almost
   certainly use entirely different models/checkpoints — get the actual
   file names expected under `/model-cache/{service_name}/` from the user
   or the existing model-loading code in that repo, don't guess filenames.
6. **Chunk length consistency across services.** `video_ai_service` chunks
   in fixed 5-second windows (`CHUNK_LENGTH_SECONDS = 5` in `worker.py`, and
   `chunk_index`/`segment_start`/`segment_end` in `detection_chunks` are
   assumed pre-created using that same convention). If `lip_sync` or
   `scene_detection` naturally chunk at a different interval, the shared
   `detection_chunks` row model breaks — this needs to be resolved with the
   user, not silently changed per repo.
7. **AWS S3 / `url_source` (yt-dlp) download path** — `video_ai_service`
   downloads from either an S3 `file_key` or a `url_source` via yt-dlp, same
   bucket/creds convention as before. Confirm this applies identically to
   the other two services (probably yes, since it's the same shared source
   file) rather than assuming.

---

## 5. Full Q&A log from the `video_ai_service` session

**Q: How is the job record structured in Postgres — shared table or
separate per service?**
A: Gave the actual schema — one shared `detection_requests` table with
per-service status/score columns, plus a shared `detection_chunks` table.
(See section 3.)

**Q: Replace the old FastAPI/Celery/SQLAlchemy stack, or keep it alongside
the new worker?**
A: Replace it entirely.

**Q: XCLIP is loaded via `.from_pretrained("microsoft/xclip-base-patch16")`,
which hits Hugging Face Hub at runtime. AMI is assumed offline — what
should happen?**
A: Load XCLIP from the local AMI path too (`/model-cache/{service}/xclip-base-patch16`,
`local_files_only=True`).

**Q: If inference/processing itself fails (not the webhook), what happens
to the SQS message?**
A: Retry in-process up to 2 times for transient errors (e.g. CUDA OOM), then
mark the job failed in Postgres, send a `status=failed` webhook, and delete
the SQS message either way. Don't rely on SQS redrive/DLQ for inference
failures.

**Q (follow-up on chunk rows): `detection_chunks` has no unique constraint
on `(detection_request_id, chunk_index)` — who creates rows, and how do 3
independent services avoid racing to `INSERT` the same chunk?**
A (after a clarifying back-and-forth): rows are updated, not inserted — "yes,
update the existing rows, no need to add a unique constraint." Confirms rows
are pre-created by something upstream (not by these 3 workers).

**Q: How should a service avoid clobbering other services' data in the
shared `result_data` JSONB column?**
A: Don't share `result_data` at all — each service has its own top-level
result column instead. (User then gave the actual columns:
`overall_ai_audio_score`, `overall_ai_video_score`, `overall_lipsync_score`.)

**Q: Where should the overall video verdict/score live, given there's no
top-level verdict/score column shown yet?**
A: There are dedicated columns — see above (`overall_ai_video_score` etc).
No dedicated verdict/explanation/segments columns were given, so
`video_ai_service` dropped the old `plainEnglishExplanation`/`verdict`
concept entirely and just writes the float score + per-chunk detail.

**Q: Should the worker touch the aggregate `detection_requests.status`
column?**
A: No — only update its own `*_status` column, never the aggregate.

**Q: What should `IDLE_TIMEOUT_SECONDS` control?**
A: Just a heartbeat log, no functional effect.

**Q: What happens if the lock-file wait for a `.done` marker takes too long
or the lock holder crashed?**
A: Fixed max wait, then fail the job (no stale-lock reclaim/takeover).

**Q: Who deletes the shared downloaded source file, and when?**
A: The last service to finish, determined by checking Postgres (are all
other requested detection types on this job in a terminal status?).

---

## 6. What to hand the next agent

Paste this whole file into the new session along with: "convert `lip_sync`
(or `scene_detection`) to the same standalone-worker architecture as
`video_ai_service` — decisions in section 2 are settled, but resolve the
open questions in section 4 with me before writing code. Use
`video_ai_service`'s `worker.py`/`db.py`/`shared_storage.py`/`webhook.py` as
the structural template, adapting column names and model-loading specifics
to this repo's actual schema and models."
