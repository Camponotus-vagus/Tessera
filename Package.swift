// swift-tools-version: 6.0
// SwiftPM build of the app, used by tools/make-app.sh. The Xcode project comes from project.yml.
import PackageDescription

let package = Package(
    name: "Tessera",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "Packages/StitchKit"),
    ],
    targets: [
        .executableTarget(
            name: "Tessera",
            dependencies: [.product(name: "StitchKit", package: "StitchKit")],
            path: "App",
            exclude: ["Resources"]
        ),
    ]
)
