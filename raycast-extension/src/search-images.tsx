import {
  Action,
  ActionPanel,
  getPreferenceValues,
  Icon,
  Keyboard,
  List,
} from "@raycast/api";
import { useEffect, useState } from "react";
import { revealInFinder } from "./lib/eagle";
import {
  createIndexerClient,
  IndexerClientError,
  type SearchResultV1,
} from "./lib/indexer";
import { runUv } from "./lib/process";
import { homedir } from "os";
import { resolve } from "path";

function expandPath(p: string): string {
  return p.startsWith("~") ? resolve(homedir(), p.slice(2)) : p;
}

function formatDimensions(w: number, h: number): string {
  if (!w || !h) return "";
  return `${w}×${h}`;
}

function buildDetailMarkdown(
  item: SearchResultV1,
  thumbPath: string,
  dims: string,
): string {
  const parts: string[] = [];

  // Image takes full width at the top
  parts.push(`![${item.name}](${thumbPath})\n`);

  // Compact info line
  const infoParts: string[] = [];
  if (item.folder_name) infoParts.push(`**${item.folder_name}**`);
  if (dims) infoParts.push(dims);
  if (item.ext) infoParts.push(item.ext.toUpperCase());
  if (infoParts.length > 0) {
    parts.push(infoParts.join("  ·  "));
  }

  // Tags
  if (item.tags) {
    parts.push(
      `\n${item.tags
        .split(", ")
        .map((t) => "`" + t + "`")
        .join("  ")}`,
    );
  }

  // AI description
  if (item.ai_description) {
    parts.push(`\n---\n${item.ai_description}`);
  }

  return parts.join("\n");
}

export default function SearchEagleImages() {
  const [searchText, setSearchText] = useState("");
  const [results, setResults] = useState<SearchResultV1[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const indexerPath = expandPath(
    getPreferenceValues<{ indexerPath?: string }>().indexerPath || "",
  );

  useEffect(() => {
    const controller = new AbortController();
    setIsLoading(true);
    setError(null);

    const timer = setTimeout(async () => {
      try {
        const client = createIndexerClient({ indexerPath, run: runUv });
        const response = await client.search(searchText, 30, controller.signal);
        if (controller.signal.aborted) return;
        setResults(response.results);
        setWarnings(
          response.retrieval.semantic_available
            ? []
            : response.retrieval.warnings,
        );
      } catch (err) {
        if (controller.signal.aborted) return;
        setResults([]);
        setWarnings([]);
        setError(
          err instanceof IndexerClientError
            ? err.message
            : "Eagle Search is unavailable",
        );
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    }, 200);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [indexerPath, searchText]);

  return (
    <List
      isShowingDetail
      filtering={false}
      searchText={searchText}
      onSearchTextChange={setSearchText}
      isLoading={isLoading}
      searchBarPlaceholder="Search by concept, style, content..."
      navigationTitle={`Eagle Search${results.length > 0 ? ` (${results.length})` : ""}`}
    >
      {warnings.map((warning) => (
        <List.Item
          key={`warning:${warning}`}
          title="Semantic search unavailable"
          subtitle={warning}
          icon={Icon.ExclamationMark}
        />
      ))}
      {error ? (
        <List.EmptyView
          title="Eagle Search unavailable"
          description={error}
          icon={Icon.ExclamationMark}
        />
      ) : results.length === 0 && !isLoading ? (
        <List.EmptyView
          title="No images found"
          description={
            searchText
              ? "Try different search terms"
              : "Run ‘Index New Eagle Images’ to start the indexer"
          }
          icon={Icon.MagnifyingGlass}
        />
      ) : (
        results.map((item) => {
          const thumbPath = expandPath(item.thumbnail_path);
          const imagePath = expandPath(item.image_path);
          const dims = formatDimensions(item.width || 0, item.height || 0);

          return (
            <List.Item
              key={item.eagle_id}
              title={item.name}
              icon={{ source: thumbPath, fallback: Icon.Image }}
              quickLook={{ path: imagePath || thumbPath }}
              detail={
                <List.Item.Detail
                  markdown={buildDetailMarkdown(item, thumbPath, dims)}
                />
              }
              actions={
                <ActionPanel>
                  <Action.ToggleQuickLook title="Quick Look" />
                  <Action
                    title="Reveal in Finder"
                    icon={Icon.Finder}
                    shortcut={{ modifiers: ["cmd"], key: "return" }}
                    onAction={() => revealInFinder(imagePath || thumbPath)}
                  />
                  <Action.CopyToClipboard
                    title="Copy Image Path"
                    content={imagePath || thumbPath}
                    shortcut={{ modifiers: ["cmd"], key: "c" }}
                  />
                  <Action.Open
                    title="Open with Default App"
                    target={imagePath || thumbPath}
                    shortcut={Keyboard.Shortcut.Common.OpenWith}
                  />
                  {item.ai_description ? (
                    <Action.CopyToClipboard
                      title="Copy AI Description"
                      content={item.ai_description}
                      shortcut={{ modifiers: ["cmd", "shift"], key: "d" }}
                    />
                  ) : null}
                </ActionPanel>
              }
            />
          );
        })
      )}
    </List>
  );
}
