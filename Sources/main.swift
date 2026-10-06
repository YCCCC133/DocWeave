import Cocoa
import Foundation
import UniformTypeIdentifiers

// MARK: - 日志视图
final class LogTextView: NSTextView {
    func append(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let attr = NSAttributedString(string: text, attributes: [
                .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
                .foregroundColor: NSColor.labelColor,
            ])
            self.textStorage?.append(attr)
            self.scrollToEndOfDocument(nil)
        }
    }
    func appendColored(_ text: String, color: NSColor) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let attr = NSAttributedString(string: text, attributes: [
                .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .semibold),
                .foregroundColor: color,
            ])
            self.textStorage?.append(attr)
            self.scrollToEndOfDocument(nil)
        }
    }
}

// MARK: - 拖放区域
final class DropZoneView: NSView {
    var onFilesDropped: (([URL]) -> Void)?
    private let borderLayer = CAShapeLayer()
    private let hintLabel = NSTextField(labelWithString: "将 PDF 文件拖到这里")
    private let subLabel = NSTextField(labelWithString: "点击选择或拖入 PDF · 支持多文件 / 超大扫描版 PDF")

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.cornerRadius = 12
        layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.6).cgColor
        layer?.borderWidth = 2
        layer?.borderColor = NSColor.separatorColor.cgColor
        registerForDraggedTypes([.fileURL])

        hintLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        hintLabel.textColor = .secondaryLabelColor
        subLabel.font = .systemFont(ofSize: 11, weight: .regular)
        subLabel.textColor = .tertiaryLabelColor
        addSubview(hintLabel)
        addSubview(subLabel)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func mouseDown(with event: NSEvent) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.pdf]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.title = "选择待转换的 PDF"
        if panel.runModal() == .OK { onFilesDropped?(panel.urls) }
    }

    override func layout() {
        super.layout()
        hintLabel.sizeToFit()
        subLabel.sizeToFit()
        hintLabel.frame.origin = NSPoint(x: (bounds.width - hintLabel.frame.width)/2, y: bounds.height/2 + 10)
        subLabel.frame.origin = NSPoint(x: (bounds.width - subLabel.frame.width)/2, y: bounds.height/2 - 16)
    }

    private func highlight(_ on: Bool) {
        layer?.borderColor = (on ? NSColor.systemBlue : NSColor.separatorColor).cgColor
        layer?.backgroundColor = (on ? NSColor.controlAccentColor.withAlphaComponent(0.15) : NSColor.controlBackgroundColor.withAlphaComponent(0.6)).cgColor
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        highlight(true)
        return .copy
    }
    override func draggingExited(_ sender: NSDraggingInfo?) { highlight(false) }
    override func draggingEnded(_ sender: NSDraggingInfo) { highlight(false) }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        highlight(false)
        let objs = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true])
        let urls = (objs as? [NSURL])?.compactMap { $0 as URL } ?? []
        let pdfs = urls.filter { $0.pathExtension.lowercased() == "pdf" }
        if !pdfs.isEmpty { onFilesDropped?(pdfs) }
        return !pdfs.isEmpty
    }
}

// MARK: - 表格
final class FileListController: NSObject, NSTableViewDataSource, NSTableViewDelegate {
    var files: [URL] = []
    var onSelectionChanged: (() -> Void)?
    func tableViewSelectionDidChange(_ notification: Notification) { onSelectionChanged?() }
    func numberOfRows(in tableView: NSTableView) -> Int { files.count }
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let id = NSUserInterfaceItemIdentifier("cell")
        let cell = tableView.makeView(withIdentifier: id, owner: nil) as? NSTableCellView ?? {
            let c = NSTableCellView()
            let tf = NSTextField(labelWithString: "")
            tf.lineBreakMode = .byTruncatingMiddle
            tf.translatesAutoresizingMaskIntoConstraints = false
            c.addSubview(tf)
            c.textField = tf
            NSLayoutConstraint.activate([
                tf.leadingAnchor.constraint(equalTo: c.leadingAnchor, constant: 4),
                tf.trailingAnchor.constraint(equalTo: c.trailingAnchor, constant: -4),
                tf.centerYAnchor.constraint(equalTo: c.centerYAnchor),
            ])
            c.identifier = id
            return c
        }()
        cell.textField?.stringValue = files[row].lastPathComponent
        cell.textField?.toolTip = files[row].path
        return cell
    }
}

// MARK: - 主窗口
final class MainWindowController: NSWindowController, NSWindowDelegate {
    private let dropZone = DropZoneView()
    private let fileTable = NSTableView()
    private let fileScroll = NSScrollView()
    private let fileListController = FileListController()
    private let outField = NSTextField(string: NSHomeDirectory() + "/Desktop/MinerU输出")
    private let ocrPopup = NSPopUpButton()
    private let speedPopup = NSPopUpButton()
    private let speedHint = NSTextField(labelWithString: "平衡型：多核并行预处理 + GPU 大批量识别，正文可复制，公式转 LaTeX。")
    private let removeButton = NSButton(title: "移除选中", target: nil, action: nil)
    private let clearButton = NSButton(title: "清空队列", target: nil, action: nil)
    private let queueLabel = NSTextField(labelWithString: "待转换 0 个 · 结束后自动清空")
    private let convertButton = NSButton(title: "开始转换", target: nil, action: nil)
    private let cancelButton = NSButton(title: "取消转换", target: nil, action: nil)
    private let statusLabel = NSTextField(labelWithString: "就绪")
    private let progressBar = NSProgressIndicator()
    private let overallLabel = NSTextField(labelWithString: "整份文件 · 已保存进度 0%")
    private let stageBar = NSProgressIndicator()
    private let stageLabel = NSTextField(labelWithString: "当前阶段 · 尚未开始（与整份文件进度独立）")
    private let elapsedLabel = NSTextField(labelWithString: "转换时将显示当前文件、页段和耗时")
    private var progressTimer: Timer?
    private var startedAt: Date?
    private var currentStageText = ""
    private let logView = LogTextView()
    private let logScroll = NSScrollView()
    private var pdfs: [URL] = []
    private var isRunning = false
    private var conversionProcess: Process?
    private var cancelRequested = false
    private var skippedInputCount = 0

    convenience init() {
        self.init(window: NSWindow(contentRect: NSRect(x: 0, y: 0, width: 860, height: 760),
                                   styleMask: [.titled, .closable, .miniaturizable, .resizable],
                                   backing: .buffered, defer: false))
        guard let win = window else { return }
        win.title = "MinerU 转换器"
        win.center()
        win.setContentSize(NSSize(width: 860, height: 760))
        win.minSize = NSSize(width: 640, height: 740)
        win.delegate = self
        buildUI()
    }

    private func buildUI() {
        guard let content = window?.contentView else { return }

        // 标题
        let title = NSTextField(labelWithString: "扫描版 PDF → Markdown")
        title.font = .systemFont(ofSize: 18, weight: .bold)
        title.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(title)

        // 拖放区
        dropZone.onFilesDropped = { [weak self] urls in
            guard let self = self, !self.isRunning else { return }
            var existed = Set(self.pdfs.map { $0.path })
            for u in urls where existed.insert(u.path).inserted { self.pdfs.append(u) }
            self.refreshQueue()
            self.statusLabel.stringValue = "已添加 \(self.pdfs.count) 个文件"
            self.statusLabel.textColor = .secondaryLabelColor
        }
        dropZone.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(dropZone)

        // 文件列表
        fileTable.headerView = nil
        fileTable.dataSource = fileListController
        fileTable.delegate = fileListController
        fileTable.rowHeight = 24
        fileTable.allowsMultipleSelection = true
        fileListController.onSelectionChanged = { [weak self] in self?.updateQueueControls() }
        fileTable.usesAlternatingRowBackgroundColors = true
        fileTable.addTableColumn(NSTableColumn(identifier: NSUserInterfaceItemIdentifier("col")))
        fileScroll.documentView = fileTable
        fileScroll.hasVerticalScroller = true
        fileScroll.borderType = .bezelBorder
        fileScroll.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(fileScroll)

        // 队列管理只移除待办项，不删除原 PDF 或转换结果。
        removeButton.target = self
        removeButton.action = #selector(removeSelectedFiles)
        clearButton.target = self
        clearButton.action = #selector(clearPendingFiles)
        for button in [removeButton, clearButton] {
            button.bezelStyle = .rounded
            button.controlSize = .small
            button.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(button)
        }
        removeButton.toolTip = "从待转换队列中移除选中的文件；可按 Command 或 Shift 多选。"
        clearButton.toolTip = "清空待转换队列，不删除原文件和已保存的结果。"
        queueLabel.font = .systemFont(ofSize: 11)
        queueLabel.textColor = .secondaryLabelColor
        queueLabel.lineBreakMode = .byTruncatingTail
        queueLabel.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(queueLabel)

        // 输出目录行
        let outLabel = NSTextField(labelWithString: "输出目录:")
        outLabel.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(outLabel)
        outField.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(outField)
        let browseBtn = NSButton(title: "浏览…", target: self, action: #selector(chooseOutput))
        browseBtn.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(browseBtn)

        // OCR 模式
        let ocrLabel = NSTextField(labelWithString: "识别模式:")
        ocrLabel.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(ocrLabel)
        ocrPopup.addItems(withTitles: ["自动检测（推荐）", "强制 OCR（纯扫描件）", "仅提取文本层"])
        ocrPopup.selectItem(at: 1)
        ocrPopup.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(ocrPopup)

        let speedLabel = NSTextField(labelWithString: "转换速度:")
        speedLabel.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(speedLabel)
        speedPopup.addItems(withTitles: ["平衡型（高并发，文字 + LaTeX）", "极速型（高并发，仅 Markdown）", "极其精确型（题干增强，文字 + LaTeX）"])
        speedPopup.selectItem(at: 2)
        speedPopup.toolTip = "平衡型多核预处理与 GPU 大批量识别，保留 LaTeX；极速型多进程并发，仅输出 Markdown；极其精确型使用 PaddleOCR-VL 1.6，联合识别中文题干与行内公式。"
        speedPopup.target = self
        speedPopup.action = #selector(updateModeHint)
        speedPopup.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(speedPopup)
        speedHint.font = .systemFont(ofSize: 11)
        speedHint.textColor = .secondaryLabelColor
        speedHint.lineBreakMode = .byTruncatingTail
        speedHint.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(speedHint)

        // 转换按钮与状态
        convertButton.bezelStyle = .rounded
        convertButton.controlSize = .large
        convertButton.keyEquivalent = "\r"
        convertButton.target = self
        convertButton.action = #selector(startConvert)
        convertButton.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(convertButton)
        cancelButton.bezelStyle = .rounded
        cancelButton.controlSize = .large
        cancelButton.target = self
        cancelButton.action = #selector(stopConversion)
        cancelButton.keyEquivalent = "\u{1b}"
        cancelButton.toolTip = "取消当前整轮转换，保留已经保存的结果。"
        cancelButton.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(cancelButton)
        statusLabel.font = .systemFont(ofSize: 12, weight: .medium)
        statusLabel.lineBreakMode = .byTruncatingMiddle
        statusLabel.alignment = .center
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(statusLabel)
        progressBar.style = .bar
        progressBar.minValue = 0
        progressBar.maxValue = 100
        progressBar.isIndeterminate = false
        progressBar.translatesAutoresizingMaskIntoConstraints = false
        progressBar.toolTip = "整份文件已保存进度；只有完整识别结果落盘后才增加，不按耗时估算。"
        progressBar.setAccessibilityLabel("整份文件已保存进度")
        content.addSubview(progressBar)
        for label in [overallLabel, stageLabel] {
            label.font = .monospacedDigitSystemFont(ofSize: 11, weight: .regular)
            label.textColor = .secondaryLabelColor
            label.lineBreakMode = .byTruncatingMiddle
            label.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(label)
        }
        stageBar.style = .bar
        stageBar.minValue = 0
        stageBar.maxValue = 100
        stageBar.isIndeterminate = false
        stageBar.translatesAutoresizingMaskIntoConstraints = false
        stageBar.toolTip = "仅当前阶段的样本完成比例，会随阶段或批次重新开始；不代表整份文件完成率或剩余时间。"
        stageBar.setAccessibilityLabel("当前阶段计数，非整份文件进度")
        content.addSubview(stageBar)
        elapsedLabel.font = .monospacedDigitSystemFont(ofSize: 11, weight: .regular)
        elapsedLabel.textColor = .secondaryLabelColor
        elapsedLabel.alignment = .center
        elapsedLabel.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(elapsedLabel)

        // 日志
        logView.isEditable = false
        logView.isRichText = false
        logView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        logView.autoresizingMask = [.width]
        logScroll.documentView = logView
        logScroll.hasVerticalScroller = true
        logScroll.borderType = .bezelBorder
        logScroll.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(logScroll)

        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            title.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),

            dropZone.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 12),
            dropZone.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            dropZone.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            dropZone.heightAnchor.constraint(equalToConstant: 110),

            fileScroll.topAnchor.constraint(equalTo: dropZone.bottomAnchor, constant: 12),
            fileScroll.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            fileScroll.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            fileScroll.heightAnchor.constraint(equalToConstant: 110),

            removeButton.topAnchor.constraint(equalTo: fileScroll.bottomAnchor, constant: 6),
            removeButton.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            clearButton.centerYAnchor.constraint(equalTo: removeButton.centerYAnchor),
            clearButton.leadingAnchor.constraint(equalTo: removeButton.trailingAnchor, constant: 8),
            queueLabel.centerYAnchor.constraint(equalTo: removeButton.centerYAnchor),
            queueLabel.leadingAnchor.constraint(equalTo: clearButton.trailingAnchor, constant: 12),
            queueLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),

            outLabel.topAnchor.constraint(equalTo: removeButton.bottomAnchor, constant: 10),
            outLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            outField.centerYAnchor.constraint(equalTo: outLabel.centerYAnchor),
            outField.leadingAnchor.constraint(equalTo: outLabel.trailingAnchor, constant: 8),
            outField.trailingAnchor.constraint(equalTo: browseBtn.leadingAnchor, constant: -8),
            browseBtn.centerYAnchor.constraint(equalTo: outLabel.centerYAnchor),
            browseBtn.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),

            ocrLabel.topAnchor.constraint(equalTo: outLabel.bottomAnchor, constant: 14),
            ocrLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            ocrPopup.centerYAnchor.constraint(equalTo: ocrLabel.centerYAnchor),
            ocrPopup.leadingAnchor.constraint(equalTo: ocrLabel.trailingAnchor, constant: 8),

            speedLabel.topAnchor.constraint(equalTo: ocrLabel.bottomAnchor, constant: 14),
            speedLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            speedPopup.centerYAnchor.constraint(equalTo: speedLabel.centerYAnchor),
            speedPopup.leadingAnchor.constraint(equalTo: speedLabel.trailingAnchor, constant: 8),
            speedHint.topAnchor.constraint(equalTo: speedPopup.bottomAnchor, constant: 4),
            speedHint.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            speedHint.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            convertButton.topAnchor.constraint(equalTo: speedHint.bottomAnchor, constant: 12),
            convertButton.trailingAnchor.constraint(equalTo: content.centerXAnchor, constant: -6),
            cancelButton.leadingAnchor.constraint(equalTo: content.centerXAnchor, constant: 6),
            cancelButton.centerYAnchor.constraint(equalTo: convertButton.centerYAnchor),
            cancelButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 100),
            convertButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 140),

            statusLabel.topAnchor.constraint(equalTo: convertButton.bottomAnchor, constant: 8),
            statusLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            statusLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            overallLabel.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 8),
            overallLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            overallLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            progressBar.topAnchor.constraint(equalTo: overallLabel.bottomAnchor, constant: 4),
            progressBar.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            progressBar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            progressBar.heightAnchor.constraint(equalToConstant: 14),
            stageLabel.topAnchor.constraint(equalTo: progressBar.bottomAnchor, constant: 8),
            stageLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            stageLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            stageBar.topAnchor.constraint(equalTo: stageLabel.bottomAnchor, constant: 4),
            stageBar.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            stageBar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            stageBar.heightAnchor.constraint(equalToConstant: 10),
            elapsedLabel.topAnchor.constraint(equalTo: stageBar.bottomAnchor, constant: 6),
            elapsedLabel.centerXAnchor.constraint(equalTo: content.centerXAnchor),

            logScroll.topAnchor.constraint(equalTo: elapsedLabel.bottomAnchor, constant: 12),
            logScroll.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            logScroll.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            logScroll.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -16),
            logScroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 120),
        ])
        updateQueueControls()
        updateModeHint()
    }

    private func refreshQueue() {
        fileListController.files = pdfs
        fileTable.deselectAll(nil)
        fileTable.reloadData()
        queueLabel.stringValue = "待转换 \(pdfs.count) 个 · 结束后自动清空"
        updateQueueControls()
    }

    private func updateQueueControls() {
        removeButton.isEnabled = !isRunning && !fileTable.selectedRowIndexes.isEmpty
        clearButton.isEnabled = !isRunning && !pdfs.isEmpty
        convertButton.isEnabled = !isRunning && !pdfs.isEmpty
        cancelButton.isEnabled = isRunning && !cancelRequested
    }

    @objc private func removeSelectedFiles() {
        guard !isRunning else { return }
        let selected = fileTable.selectedRowIndexes
        guard !selected.isEmpty else { return }
        for row in selected.sorted(by: >) where pdfs.indices.contains(row) { pdfs.remove(at: row) }
        refreshQueue()
        statusLabel.stringValue = "已移除 \(selected.count) 个队列项 · 待转换 \(pdfs.count) 个"
        statusLabel.textColor = .secondaryLabelColor
    }

    @objc private func clearPendingFiles() {
        guard !isRunning else { return }
        pdfs.removeAll()
        refreshQueue()
        statusLabel.stringValue = "队列已清空，请拖入新的 PDF"
        statusLabel.textColor = .secondaryLabelColor
    }

    @objc private func chooseOutput() {
        guard !isRunning else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "选择输出目录"
        if panel.runModal() == .OK, let url = panel.url {
            outField.stringValue = url.path
        }
    }

    @objc private func updateModeHint() {
        switch speedPopup.indexOfSelectedItem {
        case 1:
            speedHint.stringValue = "极速型：多进程准备页面 + GPU 大批量识别；仅输出 Markdown，文字可复制，公式不转 LaTeX。"
        case 2:
            speedHint.stringValue = "极其精确型：PaddleOCR-VL 1.6 + MLX 批量识别；中文与行内公式一起识别，保留 LaTeX。"
        default:
            speedHint.stringValue = "平衡型：多核并行预处理 + GPU 大批量识别，正文可复制，公式转 LaTeX。"
        }
        speedHint.toolTip = speedHint.stringValue
        ocrPopup.isEnabled = !isRunning && speedPopup.indexOfSelectedItem != 1
    }

    @objc private func startConvert() {
        guard !isRunning else { return }
        guard !pdfs.isEmpty else {
            statusLabel.stringValue = "请先拖入 PDF 文件"
            statusLabel.textColor = .systemRed
            return
        }
        let outDir = outField.stringValue.trimmingCharacters(in: .whitespaces)
        guard !outDir.isEmpty else {
            statusLabel.stringValue = "请设置输出目录"
            statusLabel.textColor = .systemRed
            return
        }
        let missing = pdfs.filter { !FileManager.default.isReadableFile(atPath: $0.path) }
        skippedInputCount = missing.count
        if !missing.isEmpty {
            let paths = Set(missing.map { $0.path })
            pdfs.removeAll { paths.contains($0.path) }
            refreshQueue()
            for file in missing {
                logView.append("⚠ 已移除失效文件：\(file.path)；可能已移动、重命名或删除，请重新拖入。\n")
            }
            if pdfs.isEmpty {
                statusLabel.stringValue = "队列中的文件均已失效，请重新拖入 PDF"
                statusLabel.textColor = .systemOrange
                return
            }
        }
        isRunning = true
        cancelRequested = false
        cancelButton.title = "取消转换"
        updateQueueControls()
        outField.isEnabled = false
        ocrPopup.isEnabled = false
        speedPopup.isEnabled = false
        startedAt = Date()
        currentStageText = "正在启动解析引擎"
        progressBar.doubleValue = 0
        progressBar.isIndeterminate = false
        overallLabel.stringValue = "整份文件 · 已保存进度 0%（尚未保存页面）"
        stageLabel.stringValue = "当前阶段 · 正在启动解析引擎（尚无计数）"
        stageBar.isIndeterminate = true
        stageBar.startAnimation(nil)
        progressTimer?.invalidate()
        updateElapsed()
        progressTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            self?.updateElapsed()
        }
        statusLabel.stringValue = "正在启动解析引擎…"
        statusLabel.textColor = .systemBlue
        logView.append("\n========== 开始转换 ==========\n")

        // JSON 数组保留文件名中的逗号，避免将一个路径误拆为多个文件。
        let pdfData = try! JSONSerialization.data(withJSONObject: pdfs.map { $0.path })
        let pdfArg = String(decoding: pdfData, as: UTF8.self)
        let ocrArg: String
        switch ocrPopup.indexOfSelectedItem {
        case 1: ocrArg = "ocr"
        case 2: ocrArg = "txt"
        default: ocrArg = "auto"
        }

        let tier = ["basic", "fast", "advanced"][speedPopup.indexOfSelectedItem]
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.runConverter(pdfs: pdfArg, outDir: outDir, ocr: ocrArg, tier: tier)
        }
    }

    @objc func stopConversion() {
        guard isRunning && !cancelRequested else { return }
        cancelRequested = true
        cancelButton.isEnabled = false
        cancelButton.title = "正在取消…"
        statusLabel.stringValue = "正在取消转换，保留已保存的结果…"
        if let process = conversionProcess, process.isRunning { process.terminate() }
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        stopConversion()
        return true
    }

    private func updateElapsed() {
        guard let startedAt = startedAt else { return }
        let seconds = Int(Date().timeIntervalSince(startedAt))
        elapsedLabel.stringValue = "总耗时 \(seconds / 60) 分 \(String(format: "%02d", seconds % 60)) 秒 · 阶段计数不代表整本完成率或剩余时间"
        elapsedLabel.toolTip = elapsedLabel.stringValue
    }

    private func finishProgress(clearQueue: Bool = true) {
        progressTimer?.invalidate()
        progressTimer = nil
        progressBar.stopAnimation(nil)
        progressBar.isIndeterminate = false
        stageBar.stopAnimation(nil)
        stageBar.isIndeterminate = false
        updateElapsed()
        isRunning = false
        conversionProcess = nil
        convertButton.title = "开始转换"
        cancelButton.title = "取消转换"
        if clearQueue {
            pdfs.removeAll()
            refreshQueue()
            logView.append("待转换队列已自动清空。\n")
        } else {
            updateQueueControls()
        }
        outField.isEnabled = true
        speedPopup.isEnabled = true
        updateModeHint()
    }

    private func receiveLine(_ line: String) {
        let prefix = "[MINERU_PROGRESS]"
        if line.hasPrefix(prefix),
           let data = String(line.dropFirst(prefix.count)).data(using: .utf8),
           let event = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let message = event["message"] as? String {
            DispatchQueue.main.async {
                self.statusLabel.stringValue = message
                self.statusLabel.toolTip = message
                self.currentStageText = event["stage_message"] as? String ?? ""
                self.stageLabel.stringValue = "当前阶段 · " + (self.currentStageText.isEmpty ? message + "（尚无计数）" : self.currentStageText)
                self.stageLabel.toolTip = self.stageLabel.stringValue
                self.updateElapsed()
                let stagePercent = (event["stage_percent"] as? NSNumber)?.doubleValue
                let totalPercent = (event["percent"] as? NSNumber)?.doubleValue
                if let percent = totalPercent, percent.isFinite {
                    self.progressBar.stopAnimation(nil)
                    self.progressBar.isIndeterminate = false
                    self.progressBar.doubleValue = min(100, max(0, percent))
                    if let saved = event["saved_pages"] as? Int, let total = event["total_pages"] as? Int {
                        self.overallLabel.stringValue = String(format: "整份文件 · 已保存 %d/%d 页（%.1f%%）", saved, total, percent)
                    } else {
                        self.overallLabel.stringValue = String(format: "整份文件 · 完成进度 %.1f%%", percent)
                    }
                    self.overallLabel.toolTip = self.overallLabel.stringValue
                }
                if let percent = stagePercent, percent.isFinite {
                    self.stageBar.stopAnimation(nil)
                    self.stageBar.isIndeterminate = false
                    self.stageBar.doubleValue = min(100, max(0, percent))
                } else if self.isRunning {
                    self.stageBar.isIndeterminate = true
                    self.stageBar.startAnimation(nil)
                } else {
                    self.stageBar.stopAnimation(nil)
                    self.stageBar.isIndeterminate = false
                    self.stageBar.doubleValue = 0
                }
            }
        } else {
            logView.append(line + "\n")
        }
    }

    private func runConverter(pdfs: String, outDir: String, ocr: String, tier: String) {
        guard let resources = Bundle.main.resourceURL else {
            DispatchQueue.main.async {
                self.statusLabel.stringValue = "启动失败：找不到内置资源"
                self.statusLabel.textColor = .systemRed
                self.finishProgress(clearQueue: false)
            }
            return
        }
        let python = resources.appendingPathComponent("python/Frameworks/Python.framework/Versions/3.11/bin/python3.11")
        let script = resources.appendingPathComponent("convert.py")
        let proc = Process()
        proc.executableURL = python
        proc.arguments = ["-u", script.path, pdfs, outDir, "--ocr", ocr, "--tier", tier]
        var env = ProcessInfo.processInfo.environment
        env["MINERU_HOME"] = resources.appendingPathComponent("mineru-home").path
        env["MINERU_MODEL_BASE_DIR"] = resources.appendingPathComponent("models").path
        env["PYTHONPATH"] = resources.appendingPathComponent("venv/lib/python3.11/site-packages").path
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        DispatchQueue.main.sync { self.conversionProcess = proc }
        do {
            try proc.run()
        } catch {
            pipe.fileHandleForWriting.closeFile()
            pipe.fileHandleForReading.closeFile()
            DispatchQueue.main.async {
                self.statusLabel.stringValue = "启动失败: \(error.localizedDescription)"
                self.statusLabel.textColor = .systemRed
                self.logView.appendColored("启动失败: \(error.localizedDescription)\n", color: .systemRed)
                self.finishProgress(clearQueue: false)
            }
            return
        }
        let shouldCancel = DispatchQueue.main.sync { self.cancelRequested }
        if shouldCancel && proc.isRunning { proc.terminate() }
        // 在后台持续读管道，按完整行解析；UTF-8 汉字和 JSON 跨读取块也不会丢失。
        pipe.fileHandleForWriting.closeFile()
        var buffer = Data()
        while true {
            let data = pipe.fileHandleForReading.availableData
            if data.isEmpty { break }
            buffer.append(data)
            while let newline = buffer.firstIndex(of: 10) {
                let line = String(decoding: buffer[..<newline], as: UTF8.self)
                    .trimmingCharacters(in: .newlines)
                receiveLine(line)
                buffer.removeSubrange(...newline)
            }
        }
        if !buffer.isEmpty { receiveLine(String(decoding: buffer, as: UTF8.self)) }
        pipe.fileHandleForReading.closeFile()
        proc.waitUntilExit()
        let code = proc.terminationStatus
        DispatchQueue.main.async {
            let wasCanceled = self.cancelRequested || code == 130
            self.finishProgress()
            if wasCanceled {
                self.statusLabel.stringValue = "已取消转换，队列已清空；已保存的结果保留在输出目录"
                self.stageLabel.stringValue = "当前阶段 · 已取消；未完成区域不计入已保存进度"
                self.statusLabel.textColor = .secondaryLabelColor
                self.logView.append("\n已取消转换；请重新拖入需要转换的文件。\n")
            } else if code == 0 {
                self.progressBar.doubleValue = 100
                self.stageLabel.stringValue = "当前阶段 · 转换及保存完成"
                self.statusLabel.stringValue = self.skippedInputCount > 0
                    ? "✅ 转换完成（已跳过 \(self.skippedInputCount) 个失效文件），队列已清空"
                    : "✅ 转换完成，队列已清空"
                self.statusLabel.textColor = .systemGreen
                self.logView.appendColored("\n✅ 全部完成，已保存到 \(outDir)\n", color: .systemGreen)
                NSWorkspace.shared.open(URL(fileURLWithPath: outDir))
            } else {
                self.statusLabel.stringValue = "⚠ 转换结束，队列已清空；有失败或部分结果（见日志）"
                self.statusLabel.textColor = .systemOrange
                self.logView.appendColored("\n⚠ 转换结束（退出码 \(code)），请查看上方错误与保存位置\n", color: .systemOrange)
            }
            self.statusLabel.toolTip = self.statusLabel.stringValue
        }
    }
}

// MARK: - App
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var windowController: MainWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        windowController = MainWindowController()
        windowController?.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) {
        windowController?.stopConversion()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
