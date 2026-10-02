import Foundation

extension MatchReport {
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try encoder.encode(self)
    }

    /// The same report with file names instead of full paths, for sharing.
    public func withoutLocalPaths() -> MatchReport {
        var copy = self
        for index in copy.images.indices {
            let name = copy.images[index].name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
            copy.images[index].url = URL(string: name ?? "") ?? URL(string: "image")!
        }
        if let models = copy.configuration.learnedModels {
            copy.learnedProblem = copy.learnedProblem?.replacingOccurrences(of: models.directory.path, with: "Models")
            copy.configuration.learnedModels = LearnedModelSet(directory: URL(string: "Models")!,
                                                               keypoints: models.keypoints)
        }
        return copy
    }

    public static func decode(_ data: Data) throws -> MatchReport {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try decoder.decode(MatchReport.self, from: data)
    }
}
