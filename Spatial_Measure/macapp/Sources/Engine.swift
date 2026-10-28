import CoreVideo
import Foundation

struct EngineObject: Decodable, Identifiable {
    let id: Int
    let lengthMm: Double
    let widthMm: Double
    let box: [[Double]]
    let center: [Double]
    var areaMm2: Double?
    var outline: [[Double]]?
    var open: Bool?
}

struct ScanInfo: Decodable {
    var source: String?
    var framesSinceMarker: Int?
    var map: String?
}

struct EnginePair: Decodable {
    let a: [Double]
    let b: [Double]
    let mm: Double
}

/// One line of JSON from engine_server.py.
struct EngineMessage: Decodable {
    let type: String
    var ok: Bool?
    var message: String?
    var calibrated: Bool?
    var aiFailed: Bool?
    var error: String?
    var frameSize: [Int]?
    var markerFound: Bool?
    var markerId: Int?
    var marker: [[Double]]?
    var objects: [EngineObject]?
    var selecting: Bool?
    var scan: ScanInfo?
    var planeSource: String?
    var selectionState: String?
    var pairs: [EnginePair]?
    var distancePnpMm: Double?
    var distanceSimilarMm: Double?
    var tiltDeg: Double?
    var warnings: [String]?
    var ms: Double?
}

/// Runs the Python measurement engine as a child process and streams frames to it.
///
/// At most one frame is in flight: while the engine is busy, new camera frames are dropped
/// instead of queued, so results never fall behind the live preview.
final class EngineBridge: @unchecked Sendable {
    var onMessage: ((EngineMessage) -> Void)?
    var onExit: ((String) -> Void)?

    private var process: Process?
    private var input: FileHandle?
    private let writeQueue = DispatchQueue(label: "engine.write", qos: .userInteractive)
    private let lock = NSLock()
    private var busy = false
    private var lineBuffer = Data()
    private var lastStderr = ""
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    func start(python: String, script: String, workingDirectory: String) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = ["-u", script]
        p.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env

        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        p.standardInput = stdin
        p.standardOutput = stdout
        p.standardError = stderr
        stdout.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            if !data.isEmpty { self?.consume(data) }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            FileHandle.standardError.write(data)
            self?.lastStderr = String((self?.lastStderr ?? "") .appending(text).suffix(600))
        }
        p.terminationHandler = { [weak self] proc in
            guard let self else { return }
            self.lock.lock(); self.input = nil; self.lock.unlock()
            self.onExit?("Engine stopped (exit \(proc.terminationStatus)). \(self.lastStderr)")
        }
        try p.run()
        process = p
        input = stdin.fileHandleForWriting
    }

    func stop() {
        process?.terminate()
    }

    /// Called from the camera queue. Returns immediately; drops the frame if the engine is busy.
    func submit(_ pixelBuffer: CVPixelBuffer) {
        lock.lock()
        if busy || input == nil { lock.unlock(); return }
        busy = true
        lock.unlock()

        nonisolated(unsafe) let pixelBuffer = pixelBuffer  // retained until written; never mutated
        writeQueue.async { [self] in
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return setIdle() }
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
            var header = Data([UInt8(ascii: "F")])
            for v in [UInt32(w), UInt32(h), UInt32(bpr)] {
                withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) }
            }
            do {
                try input?.write(contentsOf: header)
                try input?.write(contentsOf: Data(bytesNoCopy: base, count: h * bpr, deallocator: .none))
            } catch {
                setIdle()
            }
        }
    }

    func send(_ kind: Character, _ json: [String: Any]) {
        guard let body = try? JSONSerialization.data(withJSONObject: json) else { return }
        writeQueue.async { [self] in
            var d = Data([kind.asciiValue!])
            withUnsafeBytes(of: UInt32(body.count).littleEndian) { d.append(contentsOf: $0) }
            d.append(body)
            try? input?.write(contentsOf: d)
        }
    }

    private func setIdle() {
        lock.lock(); busy = false; lock.unlock()
    }

    private func consume(_ data: Data) {
        lineBuffer.append(data)
        while let nl = lineBuffer.firstIndex(of: 0x0A) {
            let line = lineBuffer[lineBuffer.startIndex..<nl]
            lineBuffer = Data(lineBuffer[(nl + 1)...])
            guard let msg = try? decoder.decode(EngineMessage.self, from: line) else { continue }
            if msg.type == "result" { setIdle() }
            onMessage?(msg)
        }
    }
}
