import AppKit
import StitchKit
import SwiftUI
import UniformTypeIdentifiers

extension PanoramaFormat {
    var contentType: UTType {
        switch self {
        case .png: .png
        case .jpeg: .jpeg
        case .tiff: .tiff
        case .heic: .heic
        }
    }

    var label: String {
        switch self {
        case .png: "PNG"
        case .jpeg: "JPEG"
        case .tiff: "TIFF"
        case .heic: "HEIC"
        }
    }

    var isLossy: Bool { self == .jpeg || self == .heic }
}

/// The choices in the save panel's accessory view.
@MainActor
@Observable
final class ExportChoice {
    var options: PanoramaExportOptions
    let transparent: Bool
    let size: PixelSize
    var onFormatChange: ((PanoramaFormat) -> Void)?

    init(options: PanoramaExportOptions, transparent: Bool, size: PixelSize) {
        self.options = options
        self.transparent = transparent
        self.size = size
    }

    /// JPEG stops at 65535 pixels per side.
    func available(_ format: PanoramaFormat) -> Bool {
        format != .jpeg || max(size.width, size.height) <= 65535
    }
}

private struct ExportAccessory: View {
    @Bindable var choice: ExportChoice

    var body: some View {
        Form {
            Picker("Format", selection: Binding(get: { choice.options.format }, set: { format in
                choice.options.format = format
                choice.onFormatChange?(format)
            })) {
                ForEach(PanoramaFormat.allCases, id: \.self) {
                    Text($0.label).tag($0).disabled(!choice.available($0))
                }
            }
            // Every row stays in place, disabled when it does not apply, so the panel keeps its size.
            LabeledContent("Quality") {
                Slider(value: $choice.options.quality, in: 0.5...1, step: 0.05) {
                    Text(choice.options.quality.formatted(.percent.precision(.fractionLength(0))))
                }
            }
            .disabled(!choice.options.format.isLossy)
            Toggle("16 bits per channel", isOn: Binding(
                get: { choice.options.sixteenBit && choice.options.format.supportsSixteenBit },
                set: { choice.options.sixteenBit = $0 }
            ))
                .disabled(!choice.options.format.supportsSixteenBit)
            Text("\(choice.options.format.label) has no transparency: the empty areas become white.")
                .font(.caption).foregroundStyle(.secondary)
                .opacity(choice.transparent && !choice.options.format.supportsTransparency ? 1 : 0)
            Text("\(choice.size.width) × \(choice.size.height) px")
                .font(.caption).foregroundStyle(.secondary)
        }
        .formStyle(.columns)
        .padding(12)
        .frame(width: 380)
    }
}

@MainActor
enum ExportPanel {
    private static let defaultsKey = "panoramaExportOptions"

    /// The options of the last export, kept per user.
    static var savedOptions: PanoramaExportOptions {
        get {
            guard let data = UserDefaults.standard.data(forKey: defaultsKey),
                  let options = try? JSONDecoder().decode(PanoramaExportOptions.self, from: data) else {
                return PanoramaExportOptions(format: .jpeg, quality: 0.9)
            }
            return options
        }
        set {
            if let data = try? JSONEncoder().encode(newValue) { UserDefaults.standard.set(data, forKey: defaultsKey) }
        }
    }

    /// Shows the save panel; the chosen file and options, or nil when cancelled.
    static func run(name: String, directory: URL?, transparent: Bool, size: PixelSize) async
        -> (URL, PanoramaExportOptions)? {
        let panel = NSSavePanel()
        let choice = ExportChoice(options: savedOptions, transparent: transparent, size: size)
        if !choice.available(choice.options.format) { choice.options.format = .tiff }
        panel.allowedContentTypes = [choice.options.format.contentType]
        panel.nameFieldStringValue = name
        panel.directoryURL = directory
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        choice.onFormatChange = { [weak panel] format in panel?.allowedContentTypes = [format.contentType] }
        let accessory = NSHostingView(rootView: ExportAccessory(choice: choice))
        accessory.frame.size = accessory.fittingSize
        panel.accessoryView = accessory
        let response = if let window = NSApp.keyWindow {
            await panel.beginSheetModal(for: window)
        } else {
            await panel.begin()
        }
        guard response == .OK, let url = panel.url else { return nil }
        savedOptions = choice.options
        return (url, choice.options)
    }
}
