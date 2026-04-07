import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, 
    QSlider, QCheckBox, QSpinBox, QProgressBar, QTextEdit, 
    QFileDialog, QScrollArea
)

from inference import run_inference, get_checkpoint_info
from utils import tensor_to_pil, make_comparison


class InferenceWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    batch_progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            result = run_inference(
                self.args,
                progress_callback=lambda v: self.progress_signal.emit(v),
                batch_progress_callback=lambda v: self.batch_progress_signal.emit(v),
                log_callback=lambda m: self.log_signal.emit(m),
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished_signal.emit(None)


class CheckpointScanWorker(QThread):
    found_signal = pyqtSignal(list)

    def __init__(self, search_dirs):
        super().__init__()
        self.search_dirs = search_dirs

    def run(self):
        found = []
        for d in self.search_dirs:
            if not os.path.isdir(d): continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".pt"):
                        fp = os.path.join(root, f)
                        try:
                            sz = os.path.getsize(fp) / 1e6
                            mt = os.path.getmtime(fp)
                            found.append((fp, mt, f"{f} [{sz:.1f} MB]"))
                        except OSError: pass
        found.sort(key=lambda x: x[1], reverse=True)
        self.found_signal.emit(found)


def tensor_to_qpixmap(tensor, max_size=512):
    if tensor is None: return QPixmap()
    pil_img = tensor_to_pil(tensor)
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.array(pil_img)
    h, w, ch = arr.shape
    qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ImagePreview(QLabel):
    def __init__(self, title="Image", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet("QLabel { background-color: #1a1a2e; border: 1px solid #333; border-radius: 4px; color: #666; }")
        self.setText(f"{title}\n(No image)")
        self._pixmap = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._do_resize)

    def set_image(self, pixmap):
        self._pixmap = pixmap
        if pixmap and not pixmap.isNull():
            self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.clear()
            self.setText(f"{self.title}\n(No image)")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(30)

    def _do_resize(self):
        if self._pixmap and not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class SliderSpinBox(QWidget):
    def __init__(self, label, min_val, max_val, default, step=1, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl = QLabel(label)
        self.lbl.setMinimumWidth(90)
        layout.addWidget(self.lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)
        self.slider.setSingleStep(step)
        
        self.spin = QSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setSingleStep(step)
        self.spin.setValue(default)

        self.slider.setStyleSheet("QSlider::groove:horizontal { height: 4px; background: #333; } QSlider::handle:horizontal { background: #5b9bd5; width: 14px; margin: -5px 0; border-radius: 7px; }")
        self.spin.setStyleSheet("QSpinBox { background: #2a2a3e; color: #ddd; border: 1px solid #444; border-radius: 3px; padding: 2px 4px; min-width: 50px; }")

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)

    def value(self): return self.spin.value()


class InferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified JPEG Restorer — Inference")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self._get_stylesheet())

        self._worker = self._scan_worker = self._result = None
        self._last_dir = str(Path.home())
        self._app_dir = os.path.dirname(os.path.abspath(__file__))
        self._output_dir = str(Path.home())

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.addWidget(self._build_left_panel(), 3)
        main_layout.addWidget(self._build_right_panel(), 5)

    def _get_stylesheet(self):
        return """
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 10px; padding-top: 14px; font-weight: bold; color: #89b4fa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #585b70; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton#run_button { background-color: #1e6640; color: #a6e3a1; border: 1px solid #3b8f5e; font-weight: bold; font-size: 13px; padding: 10px; }
            QPushButton#run_button:hover { background-color: #278550; }
            QLineEdit, QComboBox { background-color: #2a2a3e; color: #cdd6f4; border: 1px solid #585b70; border-radius: 4px; padding: 5px 8px; }
            QTextEdit { background-color: #11111b; color: #a6adc8; border: 1px solid #333; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 11px; }
            QProgressBar { background-color: #313244; border: 1px solid #585b70; border-radius: 4px; text-align: center; color: #cdd6f4; }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
        """

    def _build_left_panel(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        panel = QWidget(); layout = QVBoxLayout(panel); layout.setSpacing(10)

        # Model Group
        mg = QGroupBox("Model"); m_lay = QVBoxLayout(mg)
        self.ckpt_combo = QComboBox(); m_lay.addWidget(self.ckpt_combo)
        
        row = QHBoxLayout()
        self.ckpt_path = QLineEdit(); row.addWidget(self.ckpt_path, 1)
        self.btn_ema = QPushButton("Load EMA Weights"); self.btn_ema.clicked.connect(self._load_ema_weights); row.addWidget(self.btn_ema)
        btn_brws = QPushButton("Browse"); btn_brws.clicked.connect(self._browse_ckpt); row.addWidget(btn_brws)
        btn_ref = QPushButton("Refresh"); btn_ref.clicked.connect(self._refresh_checkpoints); row.addWidget(btn_ref)
        m_lay.addLayout(row)
        
        self.ckpt_info_label = QLabel("No checkpoint selected")
        self.ckpt_info_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        m_lay.addWidget(self.ckpt_info_label)
        
        self.use_ema = QCheckBox("Use EMA Weights (if available)")
        self.use_ema.setChecked(True); self.use_ema.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        m_lay.addWidget(self.use_ema)
        self.ckpt_combo.currentIndexChanged.connect(self._on_ckpt_combo_changed)
        layout.addWidget(mg)

        # I/O Group
        ig = QGroupBox("Input / Output"); i_lay = QVBoxLayout(ig)
        
        row_in = QHBoxLayout(); self.input_path = QLineEdit()
        btn_in_file = QPushButton("File"); btn_in_file.clicked.connect(self._browse_input_file)
        btn_in_folder = QPushButton("Folder"); btn_in_folder.clicked.connect(self._browse_input_folder)
        row_in.addWidget(self.input_path, 1)
        row_in.addWidget(btn_in_file)
        row_in.addWidget(btn_in_folder)
        i_lay.addWidget(QLabel("Input Path (File or Folder):")); i_lay.addLayout(row_in)
        
        row_out = QHBoxLayout(); self.output_dir = QLineEdit(self._output_dir)
        btn_out = QPushButton("Browse"); btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self.output_dir, 1); row_out.addWidget(btn_out)
        i_lay.addWidget(QLabel("Output Folder:")); i_lay.addLayout(row_out)
        
        self.output_name = QLineEdit("restored.png")
        i_lay.addWidget(QLabel("Output filename (Ignored if batch processing folder):")); i_lay.addWidget(self.output_name)
        layout.addWidget(ig)

        # Settings Group
        sg = QGroupBox("Restoration Settings"); s_lay = QVBoxLayout(sg)
        
        self.auto_quality = QCheckBox("Auto-detect JPEG Quality from Input File(s)")
        self.auto_quality.setChecked(True)
        s_lay.addWidget(self.auto_quality)

        self.quality = SliderSpinBox("Target Quality:", 1, 100, 75)
        self.quality.setEnabled(False) # Because auto is checked by default
        self.auto_quality.toggled.connect(lambda checked: self.quality.setEnabled(not checked))
        
        self.steps = SliderSpinBox("Steps:", 1, 50, 20)
        self.passes = SliderSpinBox("Ensemble Passes:", 1, 8, 1)
        s_lay.addWidget(self.quality); s_lay.addWidget(self.steps); s_lay.addWidget(self.passes)
        

        self.use_tta = QCheckBox("Use Test-Time Augmentation (TTA)"); self.use_tta.setChecked(True); s_lay.addWidget(self.use_tta)
        self.save_comparison = QCheckBox("Save Comparison Image"); s_lay.addWidget(self.save_comparison)
        layout.addWidget(sg)

        # Tiling Group
        tg = QGroupBox("Tiling / VRAM Optimization"); t_lay = QVBoxLayout(tg)
        self.use_tiling = QCheckBox("Enable Tiling"); self.use_tiling.setChecked(True); t_lay.addWidget(self.use_tiling)
        self.tile_size = SliderSpinBox("Tile Size:", 128, 2048, 512, step=16)
        self.overlap = SliderSpinBox("Overlap:", 0, 256, 32, step=16)
        self.batch_size = SliderSpinBox("Tile Batch Size:", 1, 32, 4)
        t_lay.addWidget(self.tile_size); t_lay.addWidget(self.overlap); t_lay.addWidget(self.batch_size)
        layout.addWidget(tg)

        # Run Section
        self.run_button = QPushButton("Run Inference"); self.run_button.setObjectName("run_button")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)

        self.batch_progress = QProgressBar(); self.batch_progress.setRange(0, 100); self.batch_progress.setFormat("Batch Progress: %p%")
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setFormat("Image Progress: %p%")
        layout.addWidget(self.batch_progress)
        layout.addWidget(self.progress)
        
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140); layout.addWidget(self.log)

        layout.addStretch(); scroll.setWidget(panel)
        return scroll

    def _build_right_panel(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Results"))

        images = QHBoxLayout()
        lc = QVBoxLayout(); lc.addWidget(QLabel("Source")); self.src_preview = ImagePreview("Source"); lc.addWidget(self.src_preview, 1)
        rc = QVBoxLayout(); rc.addWidget(QLabel("Restored")); self.pred_preview = ImagePreview("Restored"); rc.addWidget(self.pred_preview, 1)
        images.addLayout(lc); images.addLayout(rc); layout.addLayout(images, 1)

        self.comparison_check = QCheckBox("Show Comparison (Original | Restored)")
        self.comparison_check.toggled.connect(self._toggle_comparison)
        layout.addWidget(self.comparison_check)

        self.comparison_preview = ImagePreview("Comparison"); self.comparison_preview.hide(); layout.addWidget(self.comparison_preview, 1)
        return panel

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _load_ema_weights(self):
        curr_path = self.ckpt_path.text()
        if not curr_path:
            return self._log("No checkpoint selected.")
        
        dir_path = os.path.dirname(curr_path)
        ema_path = os.path.join(dir_path, "latest_ema.pt")
        
        if os.path.exists(ema_path):
            self.ckpt_path.setText(ema_path)
            self._update_ckpt_info(ema_path)
            self._log(f"Switched to EMA weights: {ema_path}")
        else:
            self._log(f"Could not find EMA weights at {ema_path}")

    def _browse_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", self._last_dir, "PyTorch Checkpoint (*.pt);;All Files (*)")
        if path: self.ckpt_path.setText(path); self._last_dir = str(Path(path).parent); self._update_ckpt_info(path)

    def _browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File", self._last_dir, "Images (*.jpg *.jpeg *.png *.webp);;All Files (*)")
        if path:
            self.input_path.setText(path)
            self._last_dir = str(Path(path).parent)
            self.output_dir.setText(self._last_dir)
            self.output_name.setText(f"{Path(path).stem}_restored.png")

    def _browse_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder for Batch Process", self._last_dir)
        if path:
            self.input_path.setText(path)
            self._last_dir = path
            self.output_dir.setText(os.path.join(path, "restored"))

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_dir.text())
        if path: self.output_dir.setText(path)

    def _refresh_checkpoints(self):
        self.ckpt_combo.blockSignals(True); self.ckpt_combo.clear(); self.ckpt_combo.addItem("Scanning...", None)
        search_dirs = [self._app_dir, os.path.join(self._app_dir, "runs"), os.path.join(self._app_dir, "..", "runs")]
        self._scan_worker = CheckpointScanWorker(search_dirs)
        self._scan_worker.found_signal.connect(self._on_checkpoints_found); self._scan_worker.start()

    def _on_checkpoints_found(self, found):
        self.ckpt_combo.clear()
        for fp, _, label in found: self.ckpt_combo.addItem(label, fp)
        if found:
            self.ckpt_combo.setCurrentIndex(0); self.ckpt_path.setText(found[0][0]); self._update_ckpt_info(found[0][0])

    def _on_ckpt_combo_changed(self, idx):
        if idx >= 0: self.ckpt_path.setText(self.ckpt_combo.itemData(idx)); self._update_ckpt_info(self.ckpt_path.text())

    def _update_ckpt_info(self, path):
        info = get_checkpoint_info(path)
        if info:
            is_ema_only = "ema" in os.path.basename(path).lower()
            if is_ema_only:
                self.ckpt_info_label.setText(
                    f"Step: {info['step']} | Arch: {info['base_channels']}ch, {info['depth']}dp | Using EMA weights"
                )
                if hasattr(self, 'btn_ema'):
                    self.btn_ema.setEnabled(False)
                self.use_ema.setEnabled(False)
                self.use_ema.setChecked(True)
                self.use_ema.setText("Using EMA weights")
            else:
                self.ckpt_info_label.setText(
                    f"Step: {info['step']} | Arch: {info['base_channels']}ch, {info['depth']}dp | EMA: {'Yes' if info['has_ema'] else 'No'}"
                )
                if hasattr(self, 'btn_ema'):
                    self.btn_ema.setEnabled(True)
                self.use_ema.setEnabled(info["has_ema"])
                self.use_ema.setChecked(info["has_ema"])
                self.use_ema.setText("Use EMA Weights (if available)")

    def _toggle_comparison(self, checked):
        if checked and self._result:
            comp_pm = tensor_to_qpixmap(make_comparison(self._result["src"], self._result["pred"]), max_size=1024)
            self.comparison_preview.set_image(comp_pm); self.comparison_preview.show()
        else: self.comparison_preview.hide()

    def _run(self):
        if not os.path.isfile(self.ckpt_path.text()): return self._log("Error: Invalid checkpoint.")
        if not os.path.exists(self.input_path.text()): return self._log("Error: Invalid input path.")

        args = {
            "weights": self.ckpt_path.text(), 
            "input": self.input_path.text(),
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "use_ema": self.use_ema.isChecked(), 
            "quality": self.quality.value(),
            "auto_quality": self.auto_quality.isChecked(),
            "steps": self.steps.value(),
            "tile": self.tile_size.value() if self.use_tiling.isChecked() else 0,
            "overlap": self.overlap.value(), 
            "batch_size": self.batch_size.value(),
            "passes": self.passes.value(), 
            "tta": self.use_tta.isChecked(),
            "save_comparison": self.save_comparison.isChecked()
        }

        self.run_button.setEnabled(False)
        self.progress.setValue(0)
        self.batch_progress.setValue(0)
        self.log.clear()
        self._log("Starting inference...")

        self._worker = InferenceWorker(args)
        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self.progress.setValue)
        self._worker.batch_progress_signal.connect(self.batch_progress.setValue)
        self._worker.error_signal.connect(lambda e: self._log(f"Error: {e}"))
        self._worker.finished_signal.connect(self._on_inference_finished)
        self._worker.start()

    def _on_inference_finished(self, result):
        self.run_button.setEnabled(True)
        if result:
            self._result = result
            self.src_preview.set_image(tensor_to_qpixmap(result["src"]))
            self.pred_preview.set_image(tensor_to_qpixmap(result["pred"]))
            self._toggle_comparison(self.comparison_check.isChecked())
            self._log("Inference completed successfully.")
            self.progress.setValue(100)
            self.batch_progress.setValue(100)
        else:
            self.progress.setValue(0)
            self.batch_progress.setValue(0)