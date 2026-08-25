import { Color, Detail, getPreferenceValues } from "@raycast/api";
import { useEffect, useState } from "react";
import { homedir } from "os";
import { resolve } from "path";

import { createIndexerClient, IndexerClientError, type RunStatusV1 } from "./lib/indexer";
import { runUv } from "./lib/process";

function expandPath(path: string): string {
  return path.startsWith("~") ? resolve(homedir(), path.slice(2)) : path;
}

export default function EagleSearchStatus() {
  const rawPath = getPreferenceValues<{ indexerPath?: string }>().indexerPath || "";
  const [status, setStatus] = useState<RunStatusV1 | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    try {
      const client = createIndexerClient({ indexerPath: expandPath(rawPath), run: runUv });
      client
        .status(controller.signal)
        .then(setStatus)
        .catch((reason: unknown) => setError(reason instanceof IndexerClientError ? reason.message : "Eagle Search is unavailable"));
    } catch (reason) {
      setError(reason instanceof IndexerClientError ? reason.message : "Eagle Search is unavailable");
    }
    return () => controller.abort();
  }, [rawPath]);

  if (error) return <Detail markdown={`# Eagle Search unavailable\n\n${error}`} />;
  if (!status) return <Detail isLoading markdown="# Checking Eagle Search status" />;
  const stateColour = status.state === "failed" || status.state === "blocked" ? Color.Red : Color.Green;
  return (
    <Detail
      metadata={
        <Detail.Metadata>
          <Detail.Metadata.TagList title="State">
            <Detail.Metadata.TagList.Item text={status.state} color={stateColour} />
          </Detail.Metadata.TagList>
          <Detail.Metadata.Label title="Stage" text={status.stage} />
          <Detail.Metadata.Label title="Progress" text={`${status.completed}/${status.total} complete; ${status.pending} pending; ${status.failed} failed`} />
          <Detail.Metadata.Label title="Caption model" text={[status.provider, status.model].filter(Boolean).join(" · ") || "Not selected"} />
          <Detail.Metadata.Label title="Semantic search" text={status.semantic_available ? "Available" : "Degraded to lexical search"} />
          {status.last_error ? <Detail.Metadata.Label title="Last error" text={status.last_error} /> : null}
        </Detail.Metadata>
      }
      markdown="# Eagle Search Status"
    />
  );
}
