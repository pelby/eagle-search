# Eagle Search

Private, local-first visual search for an Eagle image library. Eagle Search describes new images through the authenticated Codex CLI, indexes existing metadata and generated-image prompts, embeds only the resulting text locally, and exposes one versioned search contract to Raycast or future clients.

## What changed in v2

- No Gemini, OpenRouter, provider key, or per-image API billing path.
- Vision captions run through the locally logged-in `codex exec` session. Subscription usage limits still apply.
- The default caption tier is `gpt-5.6-sol` at low effort.
- Immutable caption receipts make SQLite a rebuildable projection rather than the only copy of a caption.
- Search fuses weighted SQLite FTS5 and local Nomic text embeddings with reciprocal-rank fusion. It degrades to lexical search if Ollama is unavailable.
- Original prompts embedded in image metadata are indexed directly. An embedded visual description skips the caption model.
- Human Eagle Notes are preserved byte-for-byte. Managed caption blocks are opt-in, diffable, read-back verified, and never written by ordinary indexing.
- Generated-image imports persist intent and prompt metadata before calling Eagle, reconcile ambiguous timeouts, and refuse competing import authorities.
- Raycast talks only to the Python CLI; it does not implement its own SQLite query or ranking logic.

## Architecture

```text
Eagle local API ──metadata/thumbnails──▶ indexer
generated images ──prompt sidecars─────▶ durable import queue
Codex CLI ──structured vision caption─▶ immutable receipt store
embedded prompt/description───────────▶ search projection
                                      │
                                      ▼
                       SQLite FTS5 + local Nomic embeddings
                                      │
                                      ▼
                           versioned CLI JSON contract
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                      Raycast                  future clients
```

State lives under `~/.eagle-search/` by default:

```text
db.sqlite                 derived search/index/job state
captions/                 immutable caption receipts and active pointers
thumbnails/               cached Eagle thumbnails
import-intents/           durable generated-image prompt/source metadata
run-status.json           atomic background status
evals/                    private model-evaluation fixtures and receipts
```

## Prerequisites

- Eagle v4 running while indexing or importing.
- Raycast on macOS.
- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- Codex CLI logged in with ChatGPT for caption generation.
- Ollama with `nomic-embed-text:v1.5` for semantic retrieval. Search remains usable without it.
- Filesystem permission to read the active Eagle library and generated-image folder.

No `OPENROUTER_API_KEY` or OpenAI API key is used by the default implementation.

## Install and verify

```bash
git clone https://github.com/pelby/eagle-search.git
cd eagle-search/indexer
uv sync
python3 -m unittest discover -s tests -v

cd ../raycast-extension
npm ci
npm run lint
npm run build
npm test
```

If Raycast does not infer the indexer location, set the extension's `Indexer Path` preference to the repository's absolute `indexer` directory.

## Everyday use

Open Eagle, then use **Index New Eagle Images** in Raycast. The command launches this detached worker and returns immediately:

```bash
cd eagle-search/indexer
uv run python -m src index --format jsonl
```

Indexing:

1. exports every legacy description to an immutable receipt before migrating a v1 database;
2. reads Eagle metadata and embedded prompt/description fields;
3. captions only missing or changed images;
4. stores the structured caption receipt before projecting it into SQLite;
5. embeds only the combined searchable text locally.

Search from Raycast with **Search Eagle Images**, or from the terminal:

```bash
uv run python -m src search "classroom" --mode automatic --limit 30 --json
uv run python -m src status --json
uv run python -m src stats --json
uv run python -m src embed --missing --json
uv run python -m src retry-failed --json
```

`automatic` search fuses lexical and semantic results when Ollama is healthy. `exact` is FTS-only. `best` requires semantic retrieval and reports an error instead of silently degrading.

## Eagle Notes

Normal indexing never modifies Notes. Preview the exact managed-block change first:

```bash
uv run python -m src notes-sync --dry-run --id EAGLE_ITEM_ID --json
```

Only after closing active Eagle editors and checking the diff:

```bash
uv run python -m src notes-sync --apply --quiescent --id EAGLE_ITEM_ID --json
```

The merge updates only the versioned `eagle-search:caption` block. It refuses malformed or duplicated markers, concurrent edits, and read-back mismatches.

## Generated-image prompt preservation

Import one image with an explicit prompt:

```bash
uv run python -m src import-generated \
  --path /absolute/path/image.png \
  --prompt-file /absolute/path/prompt.txt \
  --source imagegen \
  --tag ai-generated \
  --json
```

For polling, place optional sidecars beside images as `image.png.eagle-search.json`:

```json
{"prompt":"original generation prompt","source":"imagegen","tags":["ai-generated"]}
```

Then run a bounded poll or a long-lived worker:

```bash
uv run python -m src watch-generated --folder ~/Pictures/generated --once --json
uv run python -m src watch-generated --folder ~/Pictures/generated --json
```

Disable or retarget Eagle's own auto-import for the same folder first. The watcher deliberately refuses to enqueue or import while both systems could own the folder.

## Rebuild and recovery

SQLite is disposable. Rebuild a separate database from Eagle metadata and immutable receipts without caption-model calls:

```bash
uv run python -m src rebuild \
  --output ~/.eagle-search/rebuilds/db.sqlite \
  --json
```

The command never overwrites the active database or an existing destination. `--allow-notes-fallback` separately enables strict recovery from a well-formed managed Notes block when its immutable receipt is unavailable; those rows are marked `degraded-notes`.

## Caption-model evaluation

Evaluation artifacts must stay below `~/.eagle-search/evals`. The harness freezes a deterministic Stage A/B/C fixture, resumes by `(fixture, model, prompt version, image hash)`, keeps candidate corpora isolated, and selects the smallest model only after absolute quality and paired non-inferiority gates pass.

```bash
uv run python -m src eval-manifest --help
uv run python -m src eval-captions --help
uv run python -m src eval-score --help
uv run python -m src eval-report --help
uv run python -m src eval-captions-c --help
uv run python -m src eval-score-c --help
uv run python -m src eval-report-c --help
```

`eval-captions` is deliberately limited to Stages A and B. Stage C captions must run through `eval-captions-c`, which requires candidate-specific powered effects, exact gates, and hash-bound approval before it derives the hidden corpus. Final scoring also requires the blind-pool hash, the v2 pooled-relevance artifact and its packet-evidence hash. A positive final report is emitted atomically by `eval-score` or `eval-score-c` while those commands validate and score raw receipts; the standalone report commands intentionally handle preliminary results only.

The v2 harness uses atomic pixel concepts, token-safe critical terms, a separate frozen semantic-query artifact, authenticated receipt resealing, source-pixel hash checks, blinded top-ten pooling, and strict Stage C approval seals. The anonymous report includes quality, retrieval-lane, and latency evidence, but never image paths, hashes, captions, stratum names or invented subscription-cost figures. A positive final selection remains impossible without complete packet-bound blind pooled-relevance evidence scored from immutable receipts.

## Safety and reliability properties

- One filesystem lock owns caption work.
- Caption jobs and import intents have explicit pending, claimed, retryable, complete, and ambiguous states.
- Auth/quota failures stop a caption batch globally; one damaged image does not poison later jobs.
- Search has a semantic relevance floor and explicit degradation warnings.
- Generated-image adds are never blindly retried after an ambiguous timeout.
- Rebuild writes a new file atomically and verifies `PRAGMA integrity_check`.
- The test harness covers full journeys plus mutants for Notes replacement, blank captions, lock removal, semantic-floor removal, startup-scan removal, ambiguous import acknowledgement, and Raycast/CLI divergence.

## Licence

[MIT](LICENSE)
