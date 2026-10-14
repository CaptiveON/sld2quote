// ocr.swift: on-device OCR with Apple Vision. Usage: ./ocr image.png  -> JSON list of {text, conf, x, y, w, h} in pixels (origin top-left)
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count > 1, let img = NSImage(contentsOfFile: args[1]),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("usage: ocr <image>\n".data(using: .utf8)!); exit(1)
}
let W = Double(cg.width), H = Double(cg.height)
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false
req.recognitionLanguages = ["en-GB"]
req.minimumTextHeight = 0.008
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do { try handler.perform([req]) } catch { FileHandle.standardError.write("vision error\n".data(using: .utf8)!); exit(2) }
var out: [[String: Any]] = []
for obs in req.results ?? [] {
    guard let c = obs.topCandidates(1).first else { continue }
    let b = obs.boundingBox
    out.append(["text": c.string, "conf": Double(c.confidence),
                "x": b.origin.x * W, "y": (1.0 - b.origin.y - b.height) * H, "w": b.width * W, "h": b.height * H])
}
let data = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
