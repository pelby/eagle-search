/// <reference types="@raycast/api">

/* 🚧 🚧 🚧
 * This file is auto-generated from the extension's manifest.
 * Do not modify manually. Instead, update the `package.json` file.
 * 🚧 🚧 🚧 */

/* eslint-disable @typescript-eslint/ban-types */

type ExtensionPreferences = {
  /** Database Path - Path to eagle-search SQLite database */
  "dbPath": string,
  /** Indexer Path - Path to Python indexer project */
  "indexerPath": string
}

/** Preferences accessible in all the extension's commands */
declare type Preferences = ExtensionPreferences

declare namespace Preferences {
  /** Preferences accessible in the `search-images` command */
  export type SearchImages = ExtensionPreferences & {}
  /** Preferences accessible in the `reindex` command */
  export type Reindex = ExtensionPreferences & {}
}

declare namespace Arguments {
  /** Arguments passed to the `search-images` command */
  export type SearchImages = {}
  /** Arguments passed to the `reindex` command */
  export type Reindex = {}
}

