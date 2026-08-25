/** Typed argv-only client for the canonical Eagle Search Python command surface. */

export type SearchMode = "automatic" | "exact" | "best";

export interface SearchResultV1 {
  eagle_id: string;
  name: string;
  thumbnail_path: string;
  image_path: string;
  score: number;
  matched_by: string[];
  tags?: string;
  annotation?: string;
  ai_description?: string;
  folder_name?: string;
  ext?: string;
  width?: number;
  height?: number;
  created_at?: number;
}

export interface SearchResponseV1 {
  contract_version: 1;
  ok: boolean;
  query: { text: string; limit: number };
  retrieval: {
    mode: "browse" | "exact" | "lexical" | "hybrid";
    semantic_available: boolean;
    warnings: string[];
  };
  results: SearchResultV1[];
  error?: { code: string; message: string };
}

export interface RunStatusV1 {
  contract_version: 1;
  run_id: string;
  state: "idle" | "running" | "complete" | "failed" | "blocked";
  stage: string;
  total: number;
  completed: number;
  pending: number;
  failed: number;
  provider: string;
  model: string;
  semantic_available: boolean;
  last_error: string;
}

export interface ProcessResult {
  code: number;
  stdout: string;
  stderr: string;
}

export type ProcessRunner = (
  cwd: string,
  args: string[],
  signal?: AbortSignal,
) => Promise<ProcessResult>;

export class IndexerClientError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "IndexerClientError";
    this.code = code;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringField(value: unknown, name: string): string {
  if (typeof value !== "string")
    throw new IndexerClientError(
      "invalid_response",
      `Indexer returned invalid ${name}`,
    );
  return value;
}

function numberField(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new IndexerClientError(
      "invalid_response",
      `Indexer returned invalid ${name}`,
    );
  }
  return value;
}

function stringArray(value: unknown, name: string): string[] {
  if (
    !Array.isArray(value) ||
    value.some((entry) => typeof entry !== "string")
  ) {
    throw new IndexerClientError(
      "invalid_response",
      `Indexer returned invalid ${name}`,
    );
  }
  return value;
}

function optionalString(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  return stringField(value, name);
}

function optionalNumber(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  return numberField(value, name);
}

function parseJson(stdout: string): unknown {
  try {
    return JSON.parse(stdout);
  } catch {
    throw new IndexerClientError(
      "invalid_response",
      "Eagle Search returned malformed JSON",
    );
  }
}

function errorEnvelope(
  payload: Record<string, unknown>,
): { code: string; message: string } | undefined {
  if (payload.error === undefined) return undefined;
  if (!isRecord(payload.error))
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned invalid error envelope",
    );
  return {
    code: stringField(payload.error.code, "error.code"),
    message: stringField(payload.error.message, "error.message"),
  };
}

export function parseSearchResponse(value: unknown): SearchResponseV1 {
  if (!isRecord(value))
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned an invalid search response",
    );
  if (
    value.contract_version !== 1 ||
    typeof value.ok !== "boolean" ||
    !isRecord(value.query) ||
    !isRecord(value.retrieval)
  ) {
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned an unsupported search response",
    );
  }
  const error = errorEnvelope(value);
  if (!value.ok)
    throw new IndexerClientError(
      error?.code ?? "indexer_error",
      error?.message ?? "Eagle Search could not complete the request",
    );
  const retrievalMode = stringField(value.retrieval.mode, "retrieval.mode");
  if (
    !(["browse", "exact", "lexical", "hybrid"] as string[]).includes(
      retrievalMode,
    )
  ) {
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned invalid retrieval.mode",
    );
  }
  if (
    typeof value.retrieval.semantic_available !== "boolean" ||
    !Array.isArray(value.results)
  ) {
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned invalid search fields",
    );
  }
  const results = value.results.map((entry): SearchResultV1 => {
    if (!isRecord(entry))
      throw new IndexerClientError(
        "invalid_response",
        "Indexer returned an invalid result",
      );
    return {
      eagle_id: stringField(entry.eagle_id, "result.eagle_id"),
      name: stringField(entry.name, "result.name"),
      thumbnail_path: stringField(
        entry.thumbnail_path,
        "result.thumbnail_path",
      ),
      image_path: stringField(entry.image_path, "result.image_path"),
      score: numberField(entry.score, "result.score"),
      matched_by: stringArray(entry.matched_by, "result.matched_by"),
      tags: optionalString(entry.tags, "result.tags"),
      annotation: optionalString(entry.annotation, "result.annotation"),
      ai_description: optionalString(
        entry.ai_description,
        "result.ai_description",
      ),
      folder_name: optionalString(entry.folder_name, "result.folder_name"),
      ext: optionalString(entry.ext, "result.ext"),
      width: optionalNumber(entry.width, "result.width"),
      height: optionalNumber(entry.height, "result.height"),
      created_at: optionalNumber(entry.created_at, "result.created_at"),
    };
  });
  return {
    contract_version: 1,
    ok: true,
    query: {
      text: stringField(value.query.text, "query.text"),
      limit: numberField(value.query.limit, "query.limit"),
    },
    retrieval: {
      mode: retrievalMode as SearchResponseV1["retrieval"]["mode"],
      semantic_available: value.retrieval.semantic_available,
      warnings: stringArray(value.retrieval.warnings, "retrieval.warnings"),
    },
    results,
  };
}

export function parseRunStatus(value: unknown): RunStatusV1 {
  if (!isRecord(value) || value.contract_version !== 1) {
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned an invalid status response",
    );
  }
  const state = stringField(value.state, "status.state");
  if (
    !(
      ["idle", "running", "complete", "failed", "blocked"] as string[]
    ).includes(state) ||
    typeof value.semantic_available !== "boolean"
  ) {
    throw new IndexerClientError(
      "invalid_response",
      "Indexer returned an invalid status response",
    );
  }
  return {
    contract_version: 1,
    run_id: stringField(value.run_id, "status.run_id"),
    state: state as RunStatusV1["state"],
    stage: stringField(value.stage, "status.stage"),
    total: numberField(value.total, "status.total"),
    completed: numberField(value.completed, "status.completed"),
    pending: numberField(value.pending, "status.pending"),
    failed: numberField(value.failed, "status.failed"),
    provider: stringField(value.provider, "status.provider"),
    model: stringField(value.model, "status.model"),
    semantic_available: value.semantic_available,
    last_error: stringField(value.last_error, "status.last_error"),
  };
}

export function buildSearchArgs(
  query: string,
  limit: number,
  mode: SearchMode = "automatic",
): string[] {
  if (!Number.isInteger(limit) || limit < 1 || limit > 100)
    throw new IndexerClientError(
      "invalid_request",
      "Search result limit must be between 1 and 100",
    );
  // Raycast is an interactive first-party surface, so query logging remains enabled.
  // The optional --no-log flag stays available to non-interactive callers only.
  return [
    "run",
    "python",
    "-m",
    "src",
    "search",
    query,
    "--mode",
    mode,
    "--limit",
    String(limit),
    "--json",
  ];
}

export interface IndexerClientOptions {
  indexerPath: string;
  run: ProcessRunner;
}

export function createIndexerClient({
  indexerPath,
  run,
}: IndexerClientOptions) {
  if (!indexerPath.trim())
    throw new IndexerClientError(
      "configuration",
      "Set the Indexer Path in Eagle Search preferences",
    );
  const execute = async (
    args: string[],
    signal?: AbortSignal,
  ): Promise<unknown> => {
    const result = await run(indexerPath, args, signal);
    const payload = parseJson(result.stdout);
    if (result.code !== 0 && isRecord(payload) && payload.error !== undefined)
      return payload;
    if (result.code !== 0)
      throw new IndexerClientError(
        "indexer_failed",
        "Eagle Search could not complete the request",
      );
    return payload;
  };
  return {
    search: async (query: string, limit = 30, signal?: AbortSignal) =>
      parseSearchResponse(await execute(buildSearchArgs(query, limit), signal)),
    status: async (signal?: AbortSignal) =>
      parseRunStatus(
        await execute(
          ["run", "python", "-m", "src", "status", "--json"],
          signal,
        ),
      ),
  };
}
