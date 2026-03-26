import { showHUD, getPreferenceValues } from "@raycast/api";
import { execFile } from "child_process";
import { homedir } from "os";
import { resolve } from "path";

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

  await showHUD("Indexing new Eagle images...");

  return new Promise<void>((resolvePromise) => {
    const homebrewBin = "/opt/homebrew/bin";
    const envPath = process.env.PATH || "";
    const fullPath = envPath.includes(homebrewBin)
      ? envPath
      : `${homebrewBin}:${envPath}`;

    execFile(
      `${homebrewBin}/uv`,
      ["run", "python", "-m", "src", "index"],
      {
        cwd: indexerDir,
        timeout: 600000,
        env: {
          ...process.env,
          PATH: fullPath,
          HOME: homedir(),
        },
      },
      async (error, stdout, stderr) => {
        if (error) {
          const errMsg = stderr?.trim().split("\n").pop() || error.message;
          await showHUD(`❌ Indexing failed: ${errMsg.slice(0, 100)}`);
          resolvePromise();
          return;
        }

        const output = stdout.trim();
        const newCountMatch = output.match(/Indexed (\d+) new/);
        const alreadyMatch = output.match(/Already indexed: (\d+)/);
        const newCount = newCountMatch ? parseInt(newCountMatch[1]) : 0;

        if (output.includes("Nothing to index")) {
          await showHUD("✓ All Eagle images already indexed");
        } else if (newCount > 0) {
          await showHUD(`✓ Indexed ${newCount} new image${newCount === 1 ? "" : "s"}`);
        } else if (alreadyMatch) {
          await showHUD(`✓ All ${alreadyMatch[1]} images already indexed`);
        } else {
          await showHUD("✓ Indexing complete");
        }

        resolvePromise();
      }
    );
  });
}
