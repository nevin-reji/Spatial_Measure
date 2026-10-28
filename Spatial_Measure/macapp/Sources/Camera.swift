import AppKit
import AVFoundation
import SwiftUI

/// Owns the capture session. The preview layer renders straight from the session on the GPU,
/// so the on-screen video stays at the camera's native frame rate no matter how long the
/// measurement takes; frames are only *copied out* to the engine when it is free.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    var onFrame: ((CVPixelBuffer) -> Void)?
    var onFPS: ((Double) -> Void)?
    var paused = false

    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let frameQueue = DispatchQueue(label: "camera.frames", qos: .userInteractive)
    private var lastFrame = CFAbsoluteTimeGetCurrent()
    private var lastReport = CFAbsoluteTimeGetCurrent()
    private var fps = 0.0

    static func devices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera, .deskViewCamera],
            mediaType: .video, position: .unspecified
        ).devices
    }

    static func requestAccess(_ done: @escaping @Sendable (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: done(true)
        case .notDetermined: AVCaptureDevice.requestAccess(for: .video, completionHandler: done)
        default: done(false)
        }
    }

    func use(deviceID: String?) {
        sessionQueue.async { [self] in
            session.beginConfiguration()
            session.inputs.forEach { session.removeInput($0) }
            let device = Self.devices().first { $0.uniqueID == deviceID } ?? AVCaptureDevice.default(for: .video)
            if let device, let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
                session.addInput(input)
            }
            session.sessionPreset = session.canSetSessionPreset(.hd1920x1080) ? .hd1920x1080 : .high
            if session.outputs.isEmpty {
                output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                output.alwaysDiscardsLateVideoFrames = true
                output.setSampleBufferDelegate(self, queue: frameQueue)
                if session.canAddOutput(output) { session.addOutput(output) }
            }
            session.commitConfiguration()
            if !session.isRunning { session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async { [self] in session.stopRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CFAbsoluteTimeGetCurrent()
        fps = 0.9 * fps + 0.1 / max(now - lastFrame, 1e-6)
        lastFrame = now
        if now - lastReport > 0.5 {
            lastReport = now
            onFPS?(fps)
        }
        guard !paused, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pb)
    }
}

/// Native AVCaptureVideoPreviewLayer hosted in SwiftUI.
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    var paused: Bool

    func makeNSView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        return v
    }

    func updateNSView(_ v: PreviewView, context: Context) {
        guard let c = v.previewLayer.connection else { return }
        c.isEnabled = !paused
        // Engine coordinates come from unmirrored frames, so the preview must not mirror either.
        if c.isVideoMirroringSupported {
            c.automaticallyAdjustsVideoMirroring = false
            c.isVideoMirrored = false
        }
    }
}

final class PreviewView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer = CALayer()
        layer?.backgroundColor = NSColor.black.cgColor
        previewLayer.videoGravity = .resizeAspect
        layer?.addSublayer(previewLayer)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        previewLayer.frame = bounds
        CATransaction.commit()
    }
}
