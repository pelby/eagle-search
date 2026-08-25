import { showHUD, getPreferenceValues } from "@raycast/api";
import { homedir } from "os";
import { resolve } from "path";
import { launchUvIndex } from "./lib/process";

function expandPath(p: string): string {
  return p.startsWith("~") ? resolve(homedir(), p.slice(2)) : p;
}

export default async function ReindexCommand() {
  const prefs = getPreferenceValues<{ indexerPath?: string }>();
  const rawPath = prefs.indexerPath;
  if (!rawPath) {
    await showHUD("Set 'Indexer Path' in Eagle Search extension preferences");
    return;
  }
  const indexerDir = expandPath(rawPath);

  const pid = launchUvIndex(indexerDir);
  await showHUD(
    pid
      ? "Indexing started — check Eagle Search Status"
      : "Indexing start was not acknowledged",
  );
}
