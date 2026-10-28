import AVFoundation
import SwiftUI

@main
struct SpatialMeasureApp: App {
    @StateObject private var model = AppModel()
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate

    init() { signal(SIGPIPE, SIG_IGN) }

    var body: some Scene {
        WindowGroup("Spatial Measure") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 980, minHeight: 600)
                .onAppear {
                    model.start()
                    delegate.onQuit = { [model] in model.shutdown() }
                }
        }
        .windowToolbarStyle(.unified)
        .commands {
            CommandMenu("Measure") {
                Button("Log Measurement") { model.logMeasurement() }.keyboardShortcut("l")
                Button(model.paused ? "Resume" : "Pause") { model.paused.toggle() }.keyboardShortcut("p", modifiers: [.command, .shift])
                Divider()
                Button("Choose Calibration…") { model.chooseCalibration() }.keyboardShortcut("o")
                Button("Open Project Folder") { model.openProjectFolder() }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var onQuit: (() -> Void)?
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { onQuit?() }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HStack(spacing: 0) {
            ZStack {
                Color.black
                // Both layers must use the same centred aspect-fit rect for the overlay to line up.
                if let image = model.testImage {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    CameraPreview(session: model.camera.session, paused: model.paused)
                }
                OverlayView(result: model.result)
                GeometryReader { g in
                    Color.clear.contentShape(Rectangle())
                        .onTapGesture(coordinateSpace: .local) { model.tap(at: $0, in: g.size) }
                }
            }
            .overlay(alignment: .topLeading) { StatusPill().padding(12) }
            Divider()
            InspectorView().frame(width: 310)
        }
        .toolbar {
            ToolbarItemGroup {
                Button { model.paused.toggle() } label: {
                    Label(model.paused ? "Resume" : "Pause", systemImage: model.paused ? "play.fill" : "pause.fill")
                }
                .help(model.paused ? "Resume (⇧⌘P)" : "Pause (⇧⌘P)")
                Button { model.clearSelection() } label: { Label("Clear", systemImage: "xmark.circle") }
                    .disabled(!model.selecting)
                    .keyboardShortcut(.escape, modifiers: [])
                    .help("Stop measuring the tapped object (Esc)")
                Button { model.logMeasurement() } label: { Label("Log", systemImage: "square.and.arrow.down") }
                    .help("Log current measurements to measurements.csv (⌘L)")
            }
        }
    }
}

struct StatusPill: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            Circle().fill(model.result?.markerFound == true ? .green : .orange).frame(width: 8, height: 8)
            Text(model.status).lineLimit(2)
            if model.testImage == nil { Text("Camera \(Int(model.cameraFPS.rounded())) fps").foregroundStyle(.secondary) }
            Text("Engine \(Int(model.engineFPS.rounded())) fps").foregroundStyle(.secondary)
            if let ms = model.result?.ms { Text("\(Int(ms)) ms").foregroundStyle(.secondary) }
        }
        .font(.callout.monospacedDigit())
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

/// Draws engine results over the preview, mapped from frame pixels to the aspect-fit video rect.
struct OverlayView: View {
    let result: EngineMessage?

    var body: some View {
        Canvas { ctx, size in
            guard let r = result, let fs = r.frameSize, fs.count == 2 else { return }
            let W = Double(fs[0]), H = Double(fs[1])
            let s = min(size.width / W, size.height / H)
            let ox = (size.width - W * s) / 2, oy = (size.height - H * s) / 2
            func pt(_ p: [Double]) -> CGPoint { CGPoint(x: ox + p[0] * s, y: oy + p[1] * s) }
            func poly(_ pts: [[Double]]) -> Path {
                var path = Path()
                path.addLines(pts.map(pt))
                path.closeSubpath()
                return path
            }
            func tag(_ text: String, at p0: CGPoint, color: Color) {
                let p = CGPoint(x: min(max(p0.x, 90), size.width - 90), y: min(max(p0.y, 24), size.height - 24))
                let t = ctx.resolve(Text(text).font(.system(size: 12, weight: .semibold).monospacedDigit()).foregroundColor(.white))
                let sz = t.measure(in: size)
                let rect = CGRect(x: p.x - sz.width / 2 - 6, y: p.y - sz.height / 2 - 3, width: sz.width + 12, height: sz.height + 6)
                ctx.fill(Path(roundedRect: rect, cornerRadius: rect.height / 2), with: .color(color.opacity(0.85)))
                ctx.draw(t, at: p)
            }

            if let m = r.marker, m.count == 4 {
                ctx.stroke(poly(m), with: .color(.cyan), lineWidth: 2)
                let top = m.map(pt).min { $0.y < $1.y }!
                tag("REF \(r.markerId ?? 0)", at: CGPoint(x: top.x, y: top.y - 14), color: .cyan)
            }
            for pair in r.pairs ?? [] {
                var line = Path()
                line.move(to: pt(pair.a))
                line.addLine(to: pt(pair.b))
                ctx.stroke(line, with: .color(.orange), style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
                let a = pt(pair.a), b = pt(pair.b)
                tag(String(format: "%.0f mm", pair.mm), at: CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2), color: .orange)
            }
            for o in r.objects ?? [] {
                if let outline = o.outline, outline.count >= 3 {
                    // Tapped object: its real outline, with the measured rotated box as a thin guide.
                    ctx.stroke(poly(o.box), with: .color(.yellow.opacity(0.7)), style: StrokeStyle(lineWidth: 1.2, dash: [5, 4]))
                    let path = poly(outline)
                    ctx.fill(path, with: .color(.green.opacity(0.22)))
                    ctx.stroke(path, with: .color(.green), style: StrokeStyle(lineWidth: 3, lineJoin: .round))
                    let top = o.box.map(pt).min { $0.y < $1.y }!
                    let open = o.open == true
                    tag(String(format: open ? "≥ %.0f × %.0f mm so far" : "≈ %.1f × %.1f mm", o.lengthMm, o.widthMm),
                        at: CGPoint(x: pt(o.center).x, y: top.y - 16), color: open ? .orange : .green)
                    continue
                }
                let path = poly(o.box)
                ctx.fill(path, with: .color(.green.opacity(0.12)))
                ctx.stroke(path, with: .color(.green), lineWidth: 2)
                let top = o.box.map(pt).min { $0.y < $1.y }!
                tag(String(format: "#%d  %.1f × %.1f mm", o.id, o.lengthMm, o.widthMm), at: CGPoint(x: pt(o.center).x, y: top.y - 14), color: .green)
            }
        }
        .allowsHitTesting(false)
    }
}

struct InspectorView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        Form {
            Section("Objects") {
                let objects = model.result?.objects ?? []
                if model.mode == "nomarker" && !model.selecting {
                    Label("Click the object you are holding", systemImage: "cursorarrow.click").foregroundStyle(.secondary)
                } else if model.result?.markerFound != true && !model.selecting && model.mode != "nomarker" {
                    Label("Place the ArUco marker flat in view", systemImage: "viewfinder").foregroundStyle(.secondary)
                } else if objects.isEmpty && !model.selecting {
                    Label("No objects found next to the marker", systemImage: "square.dashed").foregroundStyle(.secondary)
                } else if !model.selecting {
                    Label("Click an object in the video to outline and measure it", systemImage: "cursorarrow.click").foregroundStyle(.secondary).font(.caption)
                }
                ForEach(objects) { o in
                    if let area = o.areaMm2 {
                        let sign = o.open == true ? "≥" : "≈"
                        LabeledContent("Length", value: String(format: "\(sign) %.1f mm  (%.1f cm)", o.lengthMm, o.lengthMm / 10))
                        LabeledContent("Width", value: String(format: "\(sign) %.1f mm  (%.1f cm)", o.widthMm, o.widthMm / 10))
                        LabeledContent("Area", value: String(format: "\(sign) %.0f mm²  (%.1f cm²)", area, area / 100))
                        if o.open == true {
                            Label("Not fully seen yet — pan along the object until the ends are in view", systemImage: "arrow.left.and.right")
                                .font(.caption).foregroundStyle(.orange)
                        }
                        if let b64 = model.result?.scan?.map, let data = Data(base64Encoded: b64), let img = NSImage(data: data) {
                            Image(nsImage: img).resizable().interpolation(.none).aspectRatio(contentMode: .fit)
                                .frame(maxWidth: .infinity, maxHeight: 130)
                                .help("Scan map: green = object so far, grey = area seen")
                        }
                    } else {
                        HStack {
                            Text("#\(o.id)").foregroundStyle(.secondary)
                            Spacer()
                            Text(String(format: "%.1f × %.1f mm", o.lengthMm, o.widthMm)).monospacedDigit().fontWeight(.medium)
                        }
                    }
                }
                if model.selecting {
                    Button { model.clearSelection() } label: { Label("Clear selection (Esc)", systemImage: "xmark.circle").frame(maxWidth: .infinity) }
                }
                ForEach(Array((model.result?.pairs ?? []).enumerated()), id: \.offset) { _, p in
                    HStack {
                        Text("Spacing").foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.0f mm", p.mm)).monospacedDigit()
                    }
                }
            }

            Section("Camera") {
                LabeledContent("Distance (PnP)", value: format(model.result?.distancePnpMm, "mm"))
                LabeledContent("Distance (W·f/P)", value: format(model.result?.distanceSimilarMm, "mm"))
                LabeledContent("Plane tilt", value: format(model.result?.tiltDeg, "°"))
                if !model.calibrated {
                    Text("Distances need a calibration file.").font(.caption).foregroundStyle(.secondary)
                }
                ForEach(model.result?.warnings ?? [], id: \.self) { w in
                    Label(w, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange).font(.caption)
                }
            }

            Section("Mode") {
                Picker("Object is", selection: $model.mode) {
                    Text("Table").tag("table")
                    Text("Hand + marker").tag("hand")
                    Text("Hand, no marker").tag("nomarker")
                }
                .pickerStyle(.segmented).labelsHidden()
                Text(model.mode == "nomarker"
                     ? "No marker: hold the object at the distance set below (measure it with a ruler once) and facing the camera, then click it. Sizes are estimates — they scale directly with that distance."
                     : model.mode == "hand"
                     ? "Hold the printed marker next to the object, at the same distance from the camera and facing it, then click the object. It is followed as you move it."
                     : "Lay the marker flat beside the object, click the object, and slide the camera along it if it is bigger than the frame.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.mode == "nomarker" {
                    LabeledContent("Distance to object") {
                        HStack(spacing: 4) {
                            TextField("", value: $model.distanceCM, format: .number.precision(.fractionLength(0...1)))
                                .multilineTextAlignment(.trailing).frame(width: 60)
                            Text("cm").foregroundStyle(.secondary)
                        }
                    }
                    if !model.calibrated {
                        LabeledContent("Camera field of view") {
                            HStack(spacing: 4) {
                                TextField("", value: $model.fovDeg, format: .number.precision(.fractionLength(0...1)))
                                    .multilineTextAlignment(.trailing).frame(width: 60)
                                Text("°").foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }

            Section("Settings") {
                if model.testImage == nil {
                    Picker("Camera", selection: $model.cameraID) {
                        ForEach(model.cameras, id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) }
                    }
                }
                LabeledContent("Marker size") {
                    HStack(spacing: 4) {
                        TextField("", value: $model.markerMM, format: .number.precision(.fractionLength(0...2)))
                            .multilineTextAlignment(.trailing).frame(width: 70)
                        Text("mm").foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Calibration") {
                    HStack {
                        Text(model.calibrationPath.map { ($0 as NSString).lastPathComponent } ?? "None")
                            .lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        Button("Choose…") { model.chooseCalibration() }
                        if model.calibrationPath != nil {
                            Button { model.calibrationPath = nil } label: { Image(systemName: "xmark.circle.fill") }
                                .buttonStyle(.borderless)
                        }
                    }
                }
                if !model.inHand { Toggle("Scan large objects (follow beyond the frame)", isOn: $model.scanMode) }
                Toggle("Smart outline (AI, tracks only the object)", isOn: $model.smartOutline)
                Toggle("Smooth readings", isOn: $model.smoothing)
                Toggle("Shadow suppression (CLAHE)", isOn: $model.useCLAHE)
                Toggle("Sub-pixel edge refinement", isOn: $model.refine)
            }

            Section("Benchmark log") {
                TextField("Object label", text: $model.label, prompt: Text("Credit card"))
                TextField("Caliper size", text: $model.truthSize, prompt: Text("85.60x53.98"))
                TextField("True distance (mm)", text: $model.truthDistance, prompt: Text("600"))
                Button { model.logMeasurement() } label: {
                    Label("Log measurement", systemImage: "square.and.arrow.down").frame(maxWidth: .infinity)
                }
                .keyboardShortcut("l")
                if !model.logStatus.isEmpty {
                    Text(model.logStatus).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    private func format(_ v: Double?, _ unit: String) -> String {
        guard let v else { return "—" }
        return unit == "°" ? String(format: "%.0f°", v) : String(format: "%.0f %@", v, unit)
    }
}
