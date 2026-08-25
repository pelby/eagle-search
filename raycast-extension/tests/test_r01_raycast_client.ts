import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import {
  IndexerClientError,
  buildSearchArgs,
  createIndexerClient,
  parseSearchResponse,
} from "../src/lib/indexer.ts";

const success = {
  contract_version: 1,
  ok: true,
  query: { text: "classroom", limit: 30 },
  retrieval: { mode: "lexical", semantic_available: false, warnings: ["Ollama unavailable"] },
  results: [
    {
      eagle_id: "fixture-id",
      name: "Classroom robot",
      thumbnail_path: "/tmp/thumb.png",
      image_path: "/tmp/image.png",
      score: 0.1,
      matched_by: ["lexical"],
    },
  ],
};

test("maps the canonical SearchResponseV1 and retains degradation warnings", () => {
  const response = parseSearchResponse(success);

  assert.equal(response.results[0].eagle_id, "fixture-id");
  assert.equal(response.retrieval.semantic_available, false);
  assert.deepEqual(response.retrieval.warnings, ["Ollama unavailable"]);
});

test("uses argv only and maps an indexer error to a stable client error", async () => {
  const calls: string[][] = [];
  const client = createIndexerClient({
    indexerPath: "/private/indexer",
    run: async (_cwd, args) => {
      calls.push(args);
      return { code: 2, stdout: JSON.stringify({ ...success, ok: false, error: { code: "missing_db", message: "no db" } }), stderr: "" };
    },
  });

  await assert.rejects(client.search('"classroom"; rm -rf /'), (error: unknown) => {
    assert.ok(error instanceof IndexerClientError);
    assert.equal(error.code, "missing_db");
    assert.equal(error.message, "no db");
    return true;
  });
  assert.deepEqual(calls[0], buildSearchArgs('"classroom"; rm -rf /', 30));
  assert.equal(calls[0].includes("--no-log"), false);
  assert.throws(() => buildSearchArgs("classroom", 101), /between 1 and 100/);
});

test("contains no TypeScript SQLite search or ranking implementation", async () => {
  const files = ["src/search-images.tsx", "src/reindex.tsx", "src/lib/indexer.ts"];
  const source = (await Promise.all(files.map((file) => readFile(new URL(`../${file}`, import.meta.url), "utf8")))).join("\n");

  assert.doesNotMatch(source, /executeSQL|images_fts|\bSELECT\b|\bMATCH\b/i);
});
