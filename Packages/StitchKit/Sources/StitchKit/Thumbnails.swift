import CoreGraphics
import Foundation

public enum Thumbnails {
    /// Oriented thumbnail with the given long side, decoded off the main thread by the caller.
    public static func load(_ url: URL, longSide: Int) throws -> CGImage {
        try ImageLoader.thumbnail(url: url, longSide: longSide)
    }

    /// Size and EXIF data without decoding pixels.
    public static func describe(_ url: URL, id: Int) throws -> SourceImage {
        try ImageLoader.describe(url: url, id: id)
    }
}
