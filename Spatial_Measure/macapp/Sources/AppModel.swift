import AppKit
import AVFoundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    @Published var result: EngineMessage?
    @Published var status = "Starting…"
    @Published var engineFPS = 0.0
    @Published var cameraFPS = 0.0
    @Published var paused = false { didSet { camera.paused = paused } }
    @Published var cameras: [AVCaptureDevice] = []
    @Published var cameraID = "" { didSet { if cameraID != oldValue { camera.use(deviceID: cameraID) } } }
    @Published var testImage: NSImage?
    @Published var logStatus = ""
    @Published var calibrated = false
    @Published var selecting = false

    // Settings (persisted)
    @Published var markerMM: Double { didSet { defaults.set(markerMM, forKey: "markerMM"); pushConfig() } }
    @Published var calibrationPath: String? { didSet { defaults.set(calibrationPath, forKey: "calibrationPath"); pushConfig() } }
    @Published var useCLAHE: Bool { didSet { defaults.set(useCLAHE, forKey: "useCLAHE"); pushConfig() } }
    @Published var refine: Bool { didSet { defaults.set(refine, forKey: "refine"); pushConfig() } }
    @Published var smoothing: Bool { didSet { defaults.set(smoothing, forKey: "smoothing"); pushConfig() } }
    /// "table" | "hand" (marker beside the object) | "nomarker" (hand, scale from an assumed distance)
    @Published var mode: String { didSet { defaults.set(mode, forKey: "mode"); pushConfig(); clearSelection() } }
    @Published var distanceCM: Double { didSet { defaults.set(distanceCM, forKey: "distanceCM"); pushConfig() } }
    @Published var fovDeg: Double { didSet { defaults.set(fovDeg, forKey: "fovDeg"); pushConfig() } }
    var inHand: Bool { mode != "table" }
    @Published var smartOutline: Bool { didSet { defaults.set(smartOutline, forKey: "smartOutline"); pushConfig() } }
    @Published var scanMode: Bool { didSet { defaults.set(scanMode, forKey: "scanMode"); pushConfig() } }

    // Benchmark logging
    @Published var label = ""
    @Published var truthSize = ""
    @Published var truthDistance = ""

    let camera = CameraController()
    private let engine = EngineBridge()
    private let defaults = UserDefaults.standard
    private var lastResultTime = CFAbsoluteTimeGetCurrent()
    private var started = false
    let projectDir: String

    init() {
        projectDir = Bundle.main.object(forInfoDictionaryKey: "SMProjectDir") as? String ?? FileManager.default.currentDirectoryPath
        markerMM = defaults.object(forKey: "markerMM") as? Double ?? 50
        useCLAHE = defaults.object(forKey: "useCLAHE") as? Bool ?? true
        refine = defaults.object(forKey: "refine") as? Bool ?? true
        smoothing = defaults.object(forKey: "smoothing") as? Bool ?? true
        scanMode = defaults.object(forKey: "scanMode") as? Bool ?? true
        smartOutline = defaults.object(forKey: "smartOutline") as? Bool ?? true
        mode = defaults.string(forKey: "mode") ?? "table"
        distanceCM = defaults.object(forKey: "distanceCM") as? Double ?? 40
        fovDeg = defaults.object(forKey: "fovDeg") as? Double ?? 65
        let saved = defaults.string(forKey: "calibrationPath")
        let local = (projectDir as NSString).appendingPathComponent("calibration.npz")
        calibrationPath = saved.flatMap { FileManager.default.fileExists(atPath: $0) ? $0 : nil }
            ?? (FileManager.default.fileExists(atPath: local) ? local : nil)
    }

    func start() {
        guard !started else { return }
        started = true
        startEngine()

        if let path = ProcessInfo.processInfo.environment["MEASURE_TEST_IMAGE"] {
            startTestImage(path)
            return
        }
        camera.onFrame = { [engine] pb in engine.submit(pb) }
        camera.onFPS = { [weak self] fps in Task { @MainActor in self?.cameraFPS = fps } }
        CameraController.requestAccess { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else {
                    self.status = "Camera access denied — enable it in System Settings › Privacy & Security › Camera"
                    return
                }
                self.cameras = CameraController.devices()
                self.cameraID = self.cameras.first?.uniqueID ?? ""
                if self.cameras.isEmpty { self.status = "No camera found" }
            }
        }
    }

    func shutdown() {
        camera.stop()
        engine.stop()
    }

    private func startEngine() {
        let python = (projectDir as NSString).appendingPathComponent(".venv/bin/python")
        let script = (projectDir as NSString).appendingPathComponent("engine_server.py")
        guard FileManager.default.isExecutableFile(atPath: python) else {
            status = "Python env not found at \(python) — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
            return
        }
        engine.onMessage = { [weak self] msg in Task { @MainActor in self?.handle(msg) } }
        engine.onExit = { [weak self] text in Task { @MainActor in self?.status = text } }
        do {
            try engine.start(python: python, script: script, workingDirectory: projectDir)
            pushConfig()
        } catch {
            status = "Could not start engine: \(error.localizedDescription)"
        }
    }

    private func handle(_ msg: EngineMessage) {
        switch msg.type {
        case "ready":
            status = "Engine ready"
        case "config":
            if let c = msg.calibrated { calibrated = c }
            if msg.ok == false {
                status = msg.message ?? "Config error"
                if msg.aiFailed == true { smartOutline = false } else { calibrationPath = nil }
            }
        case "logged":
            logStatus = msg.message ?? ""
        case "result":
            let now = CFAbsoluteTimeGetCurrent()
            engineFPS = 0.85 * engineFPS + 0.15 / max(now - lastResultTime, 1e-6)
            lastResultTime = now
            if let e = msg.error { status = "Engine error: \(e)" }
            else if mode == "nomarker" {
                switch msg.selectionState {
                case "ok": status = "Estimating at \(Int(distanceCM)) cm — hold the object at that distance"
                case "nothing": status = "Couldn't outline anything there — click the object itself"
                case "lost": status = "Lost the object — hold it in view, then click it again"
                default: status = "Hold the object \(Int(distanceCM)) cm from the camera and click it"
                }
            }
            else if msg.markerFound != true && msg.planeSource == "tracked" {
                let n = msg.scan?.framesSinceMarker ?? 0
                status = "Marker out of view — following the table (\(n) frames). Keep moving along the object"
            }
            else if msg.markerFound != true {
                if msg.selecting != true { status = "Looking for the ArUco marker…" }
                else if inHand { status = "Keep the marker in view, held beside the object at the same distance" }
                else { status = "Tracking lost — move back to the marker or to a part you already scanned" }
            }
            else {
                switch msg.selectionState {
                case "no_marker": status = "Show the ArUco marker first, then tap the object"
                case "marker": status = "That's the reference marker — tap an object next to it"
                case "nothing": status = "Couldn't outline anything there — tap on the object itself"
                case "lost" where msg.selecting == true:
                    status = inHand ? "Lost the object — hold it in view, then click it again" : "Lost the object — keep it still or tap it again"
                case "ok":
                    status = inHand ? "Following the object in your hand" : (msg.objects?.first?.open == true) ? "Object continues — keep moving the camera along it" :
                        (msg.scan != nil ? "Whole object scanned" : "Measuring the tapped object")
                default: status = "Tracking — tap an object to measure it"
                }
            }
            selecting = msg.selecting == true
            result = msg
        default:
            break
        }
    }

    func pushConfig() {
        engine.send("C", [
            "marker_mm": markerMM,
            "calibration": calibrationPath as Any? ?? NSNull(),
            "use_clahe": useCLAHE,
            "refine": refine,
            "smoothing": smoothing,
            "scan": scanMode,
            "mode": mode,
            "ai": smartOutline,
            "distance_mm": distanceCM * 10,
            "hfov_deg": fovDeg,
        ])
    }

    /// A click/tap in the video area: `point` is in view coordinates; the engine wants raw frame pixels.
    func tap(at point: CGPoint, in size: CGSize) {
        guard let fs = result?.frameSize, fs.count == 2, fs[0] > 0, fs[1] > 0 else { return }
        let w = Double(fs[0]), h = Double(fs[1])
        let s = min(size.width / w, size.height / h)
        let x = (point.x - (size.width - w * s) / 2) / s
        let y = (point.y - (size.height - h * s) / 2) / s
        guard x >= 0, y >= 0, x < w, y < h else { clearSelection(); return }
        engine.send("T", ["x": x, "y": y])
        selecting = true
    }

    func clearSelection() {
        engine.send("T", ["clear": true])
        selecting = false
    }

    func logMeasurement() {
        logStatus = "Logging…"
        engine.send("L", ["label": label, "truth": truthSize, "truth_distance": truthDistance])
    }

    func chooseCalibration() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "npz")!]
        panel.directoryURL = URL(fileURLWithPath: projectDir)
        panel.message = "Choose a calibration.npz produced by calibrate.py"
        if panel.runModal() == .OK, let url = panel.url { calibrationPath = url.path }
    }

    func openProjectFolder() {
        NSWorkspace.shared.open(URL(fileURLWithPath: projectDir))
    }

    // MARK: Test mode — feeds a still image instead of the camera (MEASURE_TEST_IMAGE=path)

    private func startTestImage(_ path: String) {
        guard let image = NSImage(contentsOfFile: path), let pb = Self.pixelBuffer(from: image) else {
            status = "Could not load test image \(path)"
            return
        }
        testImage = image
        let engine = self.engine
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInteractive))
        timer.schedule(deadline: .now(), repeating: 1.0 / 30)
        timer.setEventHandler { engine.submit(pb) }
        timer.resume()
        testTimer = timer

        // MEASURE_SNAPSHOT=out.png: save the window contents after a few seconds, then quit.
        if let out = ProcessInfo.processInfo.environment["MEASURE_SNAPSHOT"] {
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                guard let view = NSApp.windows.first(where: { $0.isVisible })?.contentView,
                      let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
                view.cacheDisplay(in: view.bounds, to: rep)
                try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: out))
                NSApp.terminate(nil)
            }
        }
    }

    private var testTimer: DispatchSourceTimer?

    private static func pixelBuffer(from image: NSImage) -> CVPixelBuffer? {
        guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, cg.width, cg.height, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb), width: cg.width, height: cg.height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        ctx?.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
        return pb
    }
}
