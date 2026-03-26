import { executeSQL } from "@raycast/utils";
import { getPreferenceValues } from "@raycast/api";
import { homedir } from "os";
import { resolve } from "path";

export interface ImageResult {
  eagle_id: string;
  name: string;
  tags: string;
  annotation: string;
  ai_description: string;
  thumbnail_path: string;
  image_path: string;
  folder_name: string;
  ext: string;
  width: number;
  height: number;
  created_at: number;
  indexed_at: string;
  rank?: number;
}

interface Stats {
  total: number;
  described: number;
  last_indexed: string;
}

function getDbPath(): string {
  const prefs = getPreferenceValues<{ dbPath?: string }>();
  const raw = prefs.dbPath || "~/.eagle-search/db.sqlite";
  return raw.startsWith("~") ? resolve(homedir(), raw.slice(2)) : raw;
}

function sanitiseForSql(input: string): string {
  // Whitelist: only keep alphanumeric, spaces, hyphens, and underscores.
  // This makes SQL injection impossible regardless of quoting context.
  return input.replace(/[^a-zA-Z0-9\s\-_]/g, " ").replace(/\s+/g, " ").trim();
}

export async function searchImages(query: string, limit = 30): Promise<ImageResult[]> {
  const dbPath = getDbPath();
  const sanitised = sanitiseForSql(query);

  if (!sanitised) {
    return getAllImages(limit);
  }

  try {
    // Try FTS5 MATCH (all words must appear)
    const results = await executeSQL<ImageResult>(
      dbPath,
      `SELECT i.*, images_fts.rank
       FROM images_fts
       JOIN images i ON images_fts.rowid = i.rowid
       WHERE images_fts MATCH '${sanitised}'
       ORDER BY images_fts.rank
       LIMIT ${limit}`
    );

    if (results.length > 0) {
      return results;
    }

    // Fall back to OR match
    const words = sanitised.split(/\s+/).filter((w) => w.length > 0);
    if (words.length > 0) {
      const orQuery = words.map((w) => `"${w}"`).join(" OR ");
      const orResults = await executeSQL<ImageResult>(
        dbPath,
        `SELECT i.*, images_fts.rank
         FROM images_fts
         JOIN images i ON images_fts.rowid = i.rowid
         WHERE images_fts MATCH '${orQuery}'
         ORDER BY images_fts.rank
         LIMIT ${limit}`
      );

      if (orResults.length > 0) {
        return orResults;
      }
    }
  } catch {
    // FTS5 not available -- fall back to LIKE
  }

  // Final fallback: LIKE queries (sanitised input contains no SQL metacharacters)
  return executeSQL<ImageResult>(
    dbPath,
    `SELECT * FROM images
     WHERE name LIKE '%${sanitised}%'
        OR ai_description LIKE '%${sanitised}%'
        OR annotation LIKE '%${sanitised}%'
        OR tags LIKE '%${sanitised}%'
     ORDER BY created_at DESC
     LIMIT ${limit}`
  );
}

export async function getAllImages(limit = 50): Promise<ImageResult[]> {
  const dbPath = getDbPath();
  return executeSQL<ImageResult>(
    dbPath,
    `SELECT * FROM images ORDER BY created_at DESC LIMIT ${limit}`
  );
}

export async function getStats(): Promise<Stats> {
  const dbPath = getDbPath();
  const rows = await executeSQL<{ total: number; described: number; last_indexed: string }>(
    dbPath,
    `SELECT
       count(*) as total,
       count(CASE WHEN ai_description != '' THEN 1 END) as described,
       coalesce(max(indexed_at), 'never') as last_indexed
     FROM images`
  );
  return rows[0] || { total: 0, described: 0, last_indexed: "never" };
}
