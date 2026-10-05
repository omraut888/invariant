# ADR 0005: Immutable, content-addressed data snapshots

Status: accepted (Phase 0)

## Context
A run id hashes a data snapshot id (ADR 0002). That only makes runs reproducible if a snapshot id
always refers to exactly the same bytes. All three sources revise history in place (ADR 0004), so
re-fetching cannot reproduce an old run. The snapshot is the only durable record of the data.

## Decisions

**Layout.** `<data_dir>/snapshots/<snapshot_id>/manifest.json` plus `<source>/<table>.parquet`
(`etf/prices`, `etf/sessions`, `french/factors`, `alfred/vintages`). The manifest stores, per
source: fetch time (UTC), request params, library versions, the source's typed report (coverage,
sentinels, API counts, CRSP build). Per file it stores: path, SHA-256, row count and columns.
Tables are stored as plain columns with no index, so the Parquet round trip is exact. All dates
are `datetime64[us]`, because nanosecond resolution can't represent ALFRED's `9999-12-31`.

**Identity.** `digest = sha256(canonical_json({snapshot_schema, sources: {name: {params, files:
{path: sha256}}}}))` reuses ADR 0002's canonicalization. `snapshot_id = digest[:16]`, and the full
digest goes in the manifest. Fetch time, reports and library versions are excluded. Reports are
derived from the data, and leaving out fetch time means identical data yields the same id.

**Immutability.**
- Files are written into `.tmp-<id>-<uuid>/`, fsynced, and set to mode 0444. The directory is
  then renamed into place with a single atomic `os.rename`, so readers never see a partial
  snapshot.
- If the id already exists, the writer verifies the existing snapshot (including the full digest)
  and returns it untouched. The new fetch time is discarded and logged.
- A failed write removes its temp directory.
- `load_snapshot` validates the id format (which blocks path traversal), recomputes the digest
  from the manifest, checks that it matches the id, and re-hashes every file. Each table read
  re-hashes the exact bytes it parses. Any mismatch raises `SnapshotIntegrityError`.

## Alternatives considered
- *Hash the DataFrame contents* (e.g. `pd.util.hash_pandas_object`) instead of the Parquet bytes:
  this would be independent of serialization, but its stability across pandas versions is not
  guaranteed. It is also not what we verify on load, so a file could be altered in a way that
  hashes equal but parses differently. Rejected: we hash what we store.
- *Include fetch time in the id*: every fetch would then produce a new snapshot, even when the
  data is identical. That defeats deduplication and makes "same data" impossible to recognise.
- *Mutable `latest/` directory or overwriting by id*: this violates non-negotiable #5. Rejected.
- *Making directories read-only too*: this gives little real protection (the owner can chmod),
  and it breaks ordinary cleanup tooling. Integrity checks on load are the actual guarantee.

## Consequences
- Parquet bytes are deterministic within the locked environment, because writer options are fixed
  and the pandas schema metadata is stripped. A different pyarrow version may encode the same data
  differently and so produce a **different** id. That fails safe: an id can never refer to two
  different contents. The only cost is lost deduplication across environments. The writer's
  pandas/pyarrow versions are recorded in the manifest.
- There is no inter-process lock. Two concurrent writers of the same id race on the rename: the
  loser detects the existing snapshot, verifies it, and returns it.
- A snapshot is one coherent dataset for a run (any subset of sources). Partial refreshes create a
  new snapshot.
- Snapshots are local (`data/` is gitignored). Sharing or archiving them is out of scope for
  Phase 0.
