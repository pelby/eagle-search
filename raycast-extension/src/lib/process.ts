/** Node process adapters. The command client itself remains injectable and testable. */

import { spawn } from "child_process";

import type { ProcessResult, ProcessRunner } from "./indexer";

export const runUv: ProcessRunner = (cwd, args, signal) =>
  new Promise<ProcessResult>((resolve, reject) => {
    const child = spawn("uv", args, {
      cwd,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code: code ?? 1, stdout, stderr }));
    signal?.addEventListener("abort", () => child.kill(), { once: true });
  });

export function launchUvIndex(cwd: string): number | undefined {
  const child = spawn(
    "uv",
    ["run", "python", "-m", "src", "index", "--format", "jsonl"],
    {
      cwd,
      detached: true,
      shell: false,
      stdio: "ignore",
    },
  );
  child.on("error", () => undefined);
  child.unref();
  return child.pid;
}
