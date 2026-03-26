import { execFile } from "child_process";

export function revealInFinder(path: string): void {
  execFile("open", ["-R", path]);
}

export function openWithDefaultApp(path: string): void {
  execFile("open", [path]);
}
