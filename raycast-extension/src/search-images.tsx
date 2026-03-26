import { Action, ActionPanel, Icon, List } from "@raycast/api";
import { useState, useEffect } from "react";
import { searchImages, getAllImages, type ImageResult } from "./lib/db";
import { revealInFinder } from "./lib/eagle";
import { homedir } from "os";
import { resolve } from "path";

function expandPath(p: string): string {
  return p.startsWith("~") ? resolve(homedir(), p.slice(2)) : p;
}

function formatDimensions(w: number, h: number): string {
  if (!w || !h) return "";
  return `${w}×${h}`;
}

function buildDetailMarkdown(item: ImageResult, thumbPath: string, dims: string): string {
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
    parts.push(`\n${item.tags.split(", ").map((t) => "`" + t + "`").join("  ")}`);
  }

  // AI description
  if (item.ai_description) {
    parts.push(`\n---\n${item.ai_description}`);
  }

  return parts.join("\n");
}

export default function SearchEagleImages() {
  const [searchText, setSearchText] = useState("");
  const [results, setResults] = useState<ImageResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);

    const timer = setTimeout(async () => {
      try {
        const items = searchText.trim()
          ? await searchImages(searchText)
          : await getAllImages();
        if (!cancelled) {
          setResults(items);
        }
      } catch (err) {
        console.error("Search error:", err);
        if (!cancelled) {
          setResults([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }, 200); // debounce

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [searchText]);

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
      {results.length === 0 && !isLoading ? (
        <List.EmptyView
          title="No images found"
          description={
            searchText
              ? "Try different search terms"
              : "Run the indexer first: uv run python -m src index"
          }
          icon={Icon.MagnifyingGlass}
        />
      ) : (
        results.map((item) => {
          const thumbPath = expandPath(item.thumbnail_path);
          const imagePath = expandPath(item.image_path);
          const dims = formatDimensions(item.width, item.height);

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
                    shortcut={{ modifiers: ["cmd", "shift"], key: "o" }}
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
