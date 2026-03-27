import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QSplitter,
    QFrame,
    QScrollArea,
)

from inference import run_inference, get_checkpoint_info
from utils import tensor_to_pil, make_heatmap_3ch, make_comparison


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def tensor_to_qpixmap(tensor, max_size=512):
    if tensor is None:
        return QPixmap()
    pil_img = tensor_to_pil(tensor)
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.array(pil_img)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class InferenceWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
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
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".pt"):
                        fp = os.path.join(root, f)
                        try:
                            sz = os.path.getsize(fp) / 1e6
                            mt = os.path.getmtime(fp)
                            found.append((fp, mt, f"{f} [{sz:.1f} MB]"))
                        except OSError:
                            pass
        found.sort(key=lambda x: x[1], reverse=True)
        self.found_signal.emit(found)


class ImagePreview(QLabel):
    def __init__(self, title="Image", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet(
            "QLabel { background-color: #1a1a2e; border: 1px solid #333; border-radius: 4px; color: #666; font-size: 12px; }"
        )
        self.setText(f"{title}\n(No image)")
        self._pixmap = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._do_resize)

    def set_image(self, pixmap):
        self._pixmap = pixmap
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self.clear()
            self.setText(f"{self.title}\n(No image)")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(30)

    def _do_resize(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


class SliderSpinBox(QWidget):
    def __init__(
        self,
        label,
        min_val,
        max_val,
        default,
        step=1,
        is_float=False,
        suffix="",
        parent=None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label_widget = QLabel(label)
        self.label_widget.setMinimumWidth(90)
        self.label_widget.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(self.label_widget)

        self.is_float = is_float
        self.step = step

        if is_float:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(int(min_val / step), int(max_val / step))
            self.slider.setValue(int(default / step))
            self.spin = QDoubleSpinBox()
            self.spin.setRange(min_val, max_val)
            self.spin.setSingleStep(step)
            self.spin.setValue(default)
            self.spin.setDecimals(2)
        else:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(min_val, max_val)
            self.slider.setValue(default)
            self.spin = QSpinBox()
            self.spin.setRange(min_val, max_val)
            self.spin.setSingleStep(step)
            self.spin.setValue(default)

        if suffix:
            self.spin.setSuffix(suffix)

        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; } QSlider::handle:horizontal { background: #5b9bd5; width: 14px; margin: -5px 0; border-radius: 7px; }"
        )
        self.spin.setStyleSheet(
            "QSpinBox, QDoubleSpinBox { background: #2a2a3e; color: #ddd; border: 1px solid #444; border-radius: 3px; padding: 2px 4px; min-width: 70px; }"
        )
        self.spin.setMinimumWidth(80)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    def _on_slider_changed(self, val):
        if self.is_float:
            v = val * self.step
            self.spin.blockSignals(True)
            self.spin.setValue(v)
            self.spin.blockSignals(False)
        else:
            self.spin.blockSignals(True)
            self.spin.setValue(val)
            self.spin.blockSignals(False)

    def _on_spin_changed(self, val):
        if self.is_float:
            v = int(val / self.step)
            self.slider.blockSignals(True)
            self.slider.setValue(v)
            self.slider.blockSignals(False)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(int(val))
            self.slider.blockSignals(False)

    def value(self):
        return self.spin.value()

    def set_value(self, val):
        self.spin.setValue(val)


class InferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JPEG Restorer — Inference")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(self._get_stylesheet())

        self._worker = None
        self._scan_worker = None
        self._last_dir = str(Path.home())
        self._app_dir = os.path.dirname(os.path.abspath(__file__))
        self._output_dir = str(Path.home())

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

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
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:disabled { background-color: #1e1e2e; color: #585b70; border-color: #333; }
            QPushButton#run_button { background-color: #1e6640; color: #a6e3a1; border: 1px solid #3b8f5e; font-weight: bold; font-size: 13px; padding: 10px; }
            QPushButton#run_button:hover { background-color: #278550; }
            QPushButton#run_button:disabled { background-color: #1e1e2e; color: #585b70; border-color: #333; }
            QLineEdit, QComboBox { background-color: #2a2a3e; color: #cdd6f4; border: 1px solid #585b70; border-radius: 4px; padding: 5px 8px; }
            QLineEdit:focus, QComboBox:focus { border-color: #89b4fa; }
            QComboBox::drop-down { border: none; padding-right: 6px; }
            QComboBox QAbstractItemView { background-color: #2a2a3e; color: #cdd6f4; selection-background-color: #45475a; }
            QTextEdit { background-color: #11111b; color: #a6adc8; border: 1px solid #333; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 11px; }
            QProgressBar { background-color: #313244; border: 1px solid #585b70; border-radius: 4px; text-align: center; color: #cdd6f4; }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
            QCheckBox { color: #cdd6f4; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #585b70; border-radius: 3px; background: #2a2a3e; }
            QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
            QLabel { color: #cdd6f4; }
            QScrollArea { border: none; }
        """

    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Model selection
        model_group = QGroupBox("Model")
        mg = QVBoxLayout(model_group)

        self.ckpt_combo = QComboBox()
        self.ckpt_combo.setEditable(False)
        self.ckpt_combo.setPlaceholderText("Select checkpoint...")
        mg.addWidget(QLabel("Checkpoint:"))
        mg.addWidget(self.ckpt_combo)

        ckpt_row = QHBoxLayout()
        self.ckpt_path = QLineEdit()
        self.ckpt_path.setPlaceholderText("Path to .pt file...")
        btn_browse_ckpt = QPushButton("Browse")
        btn_browse_ckpt.clicked.connect(self._browse_ckpt)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_checkpoints)
        ckpt_row.addWidget(self.ckpt_path, 1)
        ckpt_row.addWidget(btn_browse_ckpt)
        ckpt_row.addWidget(btn_refresh)
        mg.addLayout(ckpt_row)

        self.ckpt_info_label = QLabel("No checkpoint selected")
        self.ckpt_info_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        mg.addWidget(self.ckpt_info_label)
        self.ckpt_combo.currentIndexChanged.connect(self._on_ckpt_combo_changed)

        layout.addWidget(model_group)

        # IO settings
        io_group = QGroupBox("Input / Output")
        io = QVBoxLayout(io_group)

        io.addWidget(QLabel("Input Image:"))
        input_row = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select an image file...")
        btn_input = QPushButton("Browse")
        btn_input.clicked.connect(self._browse_input)
        input_row.addWidget(self.input_path, 1)
        input_row.addWidget(btn_input)
        io.addLayout(input_row)

        io.addWidget(QLabel("Output Folder:"))
        output_row = QHBoxLayout()
        self.output_dir = QLineEdit(self._output_dir)
        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(btn_output)
        io.addLayout(output_row)

        io.addWidget(QLabel("Output filename:"))
        self.output_name = QLineEdit("restored.png")
        io.addWidget(self.output_name)

        layout.addWidget(io_group)

        # Diffusion parameters
        diff_group = QGroupBox("Diffusion Parameters")
        dg = QVBoxLayout(diff_group)

        self.quality = SliderSpinBox("Quality:", 1, 100, 75, suffix="")
        self.noise = SliderSpinBox("Noise:", 0.0, 0.5, 0.05, step=0.01, is_float=True)
        self.steps = SliderSpinBox("Steps:", 1, 8, 4)

        dg.addWidget(self.quality)
        dg.addWidget(self.noise)
        dg.addWidget(self.steps)
        layout.addWidget(diff_group)

        # Advanced
        adv_group = QGroupBox("Advanced (Ensemble)")
        ag = QVBoxLayout(adv_group)

        self.use_tiling = QCheckBox("Enable Tiling")
        self.use_tiling.setChecked(True)
        ag.addWidget(self.use_tiling)

        self.tile_size = SliderSpinBox("Tile Size:", 128, 2048, 512)
        self.overlap = SliderSpinBox("Overlap:", 0, 256, 32)
        self.passes = SliderSpinBox("Passes:", 1, 8, 1)
        self.q_jitter = SliderSpinBox(
            "Q-Jitter:", 0.0, 10.0, 2.0, step=0.1, is_float=True
        )
        self.use_tta = QCheckBox("Use TTA (Flips)")
        self.use_tta.setChecked(True)
        self.save_comparison = QCheckBox("Save Comparison Image")
        self.save_comparison.setChecked(False)
        self.use_compile = QCheckBox("Compile Model (PyTorch 2.0+)")
        self.use_compile.setChecked(False)

        ag.addWidget(self.tile_size)
        ag.addWidget(self.overlap)
        ag.addWidget(self.passes)
        ag.addWidget(self.q_jitter)
        ag.addWidget(self.use_tta)
        ag.addWidget(self.save_comparison)
        ag.addWidget(self.use_compile)
        layout.addWidget(adv_group)

        # Run
        self.run_button = QPushButton("Run Inference")
        self.run_button.setObjectName("run_button")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        self.title_label = QLabel("Results")
        self.title_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #89b4fa;"
        )
        layout.addWidget(self.title_label)

        images = QHBoxLayout()

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Source"))
        self.src_preview = ImagePreview("Source")
        left_col.addWidget(self.src_preview, 1)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Restored"))
        self.pred_preview = ImagePreview("Restored")
        right_col.addWidget(self.pred_preview, 1)

        images.addLayout(left_col)
        images.addLayout(right_col)
        layout.addLayout(images, 1)

        self.gate_check = QCheckBox("Show Gate Map (artifact heatmap)")
        self.gate_check.toggled.connect(self._toggle_gate)
        layout.addWidget(self.gate_check)

        self.gate_preview = ImagePreview("Gate Map")
        self.gate_preview.hide()
        layout.addWidget(self.gate_preview, 1)

        self.comparison_check = QCheckBox("Show Comparison (Original | Restored)")
        self.comparison_check.toggled.connect(self._toggle_comparison)
        layout.addWidget(self.comparison_check)

        self.comparison_preview = ImagePreview("Comparison")
        self.comparison_preview.hide()
        layout.addWidget(self.comparison_preview, 1)

        self._result = None
        return panel

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _browse_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint",
            self._last_dir,
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.ckpt_path.setText(path)
            self._last_dir = str(Path(path).parent)
            self._update_ckpt_info(path)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            self._last_dir,
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.input_path.setText(path)
            parent = str(Path(path).parent)
            self._last_dir = parent
            self.output_dir.setText(parent)
            stem = Path(path).stem
            self.output_name.setText(f"{stem}_restored.png")

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_dir.text()
        )
        if path:
            self.output_dir.setText(path)

    def _refresh_checkpoints(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait()

        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        self.ckpt_combo.addItem("Scanning...", None)
        self.ckpt_combo.setEnabled(False)
        self.ckpt_path.setEnabled(False)

        search_dirs = [
            self._app_dir,
            os.path.join(self._app_dir, "runs"),
            os.path.join(self._app_dir, "..", "runs"),
        ]
        self._scan_worker = CheckpointScanWorker(search_dirs)
        self._scan_worker.found_signal.connect(self._on_checkpoints_found)
        self._scan_worker.start()

    def _on_checkpoints_found(self, found):
        self.ckpt_combo.clear()
        self.ckpt_combo.setEnabled(True)
        self.ckpt_path.setEnabled(True)
        for fp, _, label in found:
            self.ckpt_combo.addItem(label, fp)
        if found:
            self.ckpt_combo.setCurrentIndex(0)
            self.ckpt_path.setText(found[0][0])
            self._update_ckpt_info(found[0][0])
        self._scan_worker = None

    def _on_ckpt_combo_changed(self, idx):
        if idx < 0:
            return
        fp = self.ckpt_combo.itemData(idx)
        if fp and os.path.isfile(fp):
            self.ckpt_path.setText(fp)
            self._update_ckpt_info(fp)

    def _update_ckpt_info(self, path):
        info = get_checkpoint_info(path)
        if info:
            cfg = info["model_config"]
            bc = cfg.get("base_channels", "?")
            ed = cfg.get("emb_dim", "?")
            dp = cfg.get("depth", "?")
            step = info["step"]
            ema = "Yes" if info["has_ema"] else "No"
            self.ckpt_info_label.setText(
                f"Diffusion Model | ch={bc} emb={ed} depth={dp} | step={step} | EMA: {ema}"
            )
        else:
            self.ckpt_info_label.setText("Could not read checkpoint")

    def _toggle_gate(self, checked):
        if checked and self._result and self._result.get("gate") is not None:
            gate_3ch = make_heatmap_3ch(self._result["gate"])
            pm = tensor_to_qpixmap(gate_3ch, max_size=512)
            self.gate_preview.set_image(pm)
            self.gate_preview.show()
        else:
            self.gate_preview.hide()

    def _toggle_comparison(self, checked):
        if checked and self._result:
            comp = make_comparison(self._result["src"], self._result["pred"])
            pm = tensor_to_qpixmap(comp, max_size=1024)
            self.comparison_preview.set_image(pm)
            self.comparison_preview.show()
        else:
            self.comparison_preview.hide()

    def _run(self):
        weights = self.ckpt_path.text().strip()
        inp = self.input_path.text().strip()
        out_dir = self.output_dir.text().strip()
        out_name = self.output_name.text().strip()

        if not weights or not os.path.isfile(weights):
            self._log("Error: Select a valid checkpoint file")
            return
        if not inp or not os.path.isfile(inp):
            self._log("Error: Select a valid input image")
            return
        if not out_dir:
            self._log("Error: Select an output directory")
            return
        if not out_name:
            self._log("Error: Enter an output filename")
            return

        tile = self.tile_size.value() if self.use_tiling.isChecked() else 0

        args = {
            "weights": weights,
            "input": inp,
            "output": os.path.join(out_dir, out_name),
            "quality": self.quality.value(),
            "noise": self.noise.value(),
            "steps": self.steps.value(),
            "tile": tile,
            "overlap": self.overlap.value(),
            "passes": self.passes.value(),
            "q_jitter": self.q_jitter.value(),
            "tta": self.use_tta.isChecked(),
            "save_comparison": self.save_comparison.isChecked(),
            "compile": self.use_compile.isChecked(),
        }

        self.log.clear()
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")

        self._worker = InferenceWorker(args)
        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self.progress.setValue)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(lambda e: self._log(f"Error: {e}"))
        self._worker.start()

    def _on_finished(self, result):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Inference")

        if result:
            self._result = result
            src_pm = tensor_to_qpixmap(result["src"], max_size=512)
            pred_pm = tensor_to_qpixmap(result["pred"], max_size=512)
            self.src_preview.set_image(src_pm)
            self.pred_preview.set_image(pred_pm)

            if self.gate_check.isChecked() and result.get("gate") is not None:
                gate_3ch = make_heatmap_3ch(result["gate"])
                gate_pm = tensor_to_qpixmap(gate_3ch, max_size=512)
                self.gate_preview.set_image(gate_pm)
                self.gate_preview.show()

            if self.comparison_check.isChecked():
                comp = make_comparison(result["src"], result["pred"])
                comp_pm = tensor_to_qpixmap(comp, max_size=1024)
                self.comparison_preview.set_image(comp_pm)
                self.comparison_preview.show()

            self._log("Done!")
        else:
            self._log("Inference failed.")

        self._worker = None

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.quit()
            self._scan_worker.wait(2000)
        event.accept()
