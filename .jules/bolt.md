# Bolt's Journal

## 2026-11-04 - Hover state re-runs SwiftUI bodies that do the heavy work
**Learning:** `GraphCanvas` and `PairCanvas` keep `hovered` in state written by `onContinuousHover`, so their `body` runs again on most mouse moves. Anything computed at the top of such a body runs at mouse rate: in `GraphCanvas` that was `GraphLayout.geometric`, which costs O(V·E) for the spanning tree plus O(V·(V+E)) for the centre search, and `evidence(filter:)` was called twice more. A report is replaced as a whole and never changed in place, so `MatchReport.createdAt` identifies it (PairView and the panorama already use it that way).
**Action:** In views that track hover or drag, keep derived data in a cache keyed by `createdAt` and the filter. A reference-type box in `@State` can be filled during `body` without the "modifying state during view update" warning. This repo can't be built on Linux, so check such changes on macOS.
