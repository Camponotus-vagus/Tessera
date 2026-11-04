# Sentinel's Journal

## 2026-11-04 - Model download trusted the server for its length
**Vulnerability:** `ModelInstaller` hashed the archive while streaming it to disk and compared the SHA-256 only at the end, with no cap on the body length, so a broken or hostile server could fill the disk before the checksum check failed.
**Learning:** The manifest (bundled `models.json`) is the trust anchor; the server is not. Anything the manifest pins (digest, size) should bound the download, not only validate it afterwards.
**Prevention:** Bound every download by the manifest's `size` when known. This repo is Swift/macOS: the StitchKit tests (`swift test` in `Packages/StitchKit`) cannot run in a Linux container.
