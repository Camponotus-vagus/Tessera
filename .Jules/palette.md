# Palette's Journal

## 2026-11-04 - Toolbar buttons show only their icon
**Learning:** In this macOS SwiftUI app, `Button("Title", systemImage:)` in the window toolbar shows only the icon, and the menu shortcuts (⌘R, ⇧⌘R, ⌘., ⌘E) appear only in the Analysis and File menus. Every user-facing string, `.help` tooltips included, needs an entry in `App/Resources/it.lproj/Localizable.strings`. English is the source language, so `en.lproj` stays empty.
**Action:** Give every icon-only toolbar control a `.help` that names its shortcut, and add the Italian key in the same change.
