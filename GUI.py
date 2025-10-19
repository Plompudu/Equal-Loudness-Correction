import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QVBoxLayout,
    QPushButton, QComboBox, QCheckBox, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# PEQ folders
# Add your Folder paths here or use the ones inside this project
# naming loosely based on https://www.audiosciencereview.com/forum/index.php?threads/reference-sound-pressure-level-flowchart.11069/
PEQ_FOLDERS = {
    "-0dB_dolby | DOLBY CINEMA": Path(r"PEQ_85dB_reference"),
    "-0dB_dolby | MUSIC 5.1 SPEAKERS": Path(r"PEQ_85dB_reference"),
    "-0dB_dolby | MUSIC STEREO HEADPHONES": Path(r"PEQ_85dB_reference"),
    "-3dB_dolby | MUSIC STEREO IEM's": Path(r"PEQ_82dB_reference"),
    "-5dB_dolby | TV HEADPHONES": Path(r"PEQ_80dB_reference"),
    "-9dB_dolby | TV SPEAKERS <1.499 ft^2": Path(r"PEQ_76dB_reference"),
    "-7dB_dolby | TV SPEAKERS 1.500 -  4.999 ft^2": Path(r"PEQ_78dB_reference"),
    "-5dB_dolby | TV SPEAKERS 5.000 -  9.999 ft^2": Path(r"PEQ_80dB_reference"),
    "-3dB_dolby | TV SPEAKERS 10.000 - 19.999 ft^2": Path(r"PEQ_82dB_reference"),
    "-0dB_dolby | TV SPEAKERS >20.000 ft^2": Path(r"PEQ_85dB_reference"),

    "85dB | 105dB peak |  -0dB_dolby Reference": Path(r"PEQ_85dB_reference"),
    "84dB | 104dB peak |  -1dB_dolby Reference": Path(r"PEQ_84dB_reference"),
    "83dB | 103dB peak |  -2dB_dolby Reference": Path(r"PEQ_83dB_reference"),
    "82dB | 102dB peak |  -3dB_dolby Reference": Path(r"PEQ_82dB_reference"),
    "81dB | 101dB peak |  -4dB_dolby Reference": Path(r"PEQ_81dB_reference"),
    "80dB | 100dB peak |  -5dB_dolby Reference": Path(r"PEQ_80dB_reference"),
    "79dB |  99dB peak |  -6dB_dolby Reference": Path(r"PEQ_79dB_reference"),
    "78dB |  98dB peak |  -7dB_dolby Reference": Path(r"PEQ_78dB_reference"),
    "77dB |  97dB peak |  -8dB_dolby Reference": Path(r"PEQ_77dB_reference"),
    "76dB |  96dB peak |  -9dB_dolby Reference": Path(r"PEQ_76dB_reference"),
    "75dB |  95dB peak | -10dB_dolby Reference": Path(r"PEQ_75dB_reference"),
    "74dB |  94dB peak | -11dB_dolby Reference": Path(r"PEQ_74dB_reference"),
    "73dB |  93dB peak | -12dB_dolby Reference": Path(r"PEQ_73dB_reference"),
    "72dB |  92dB peak | -13dB_dolby Reference": Path(r"PEQ_72dB_reference"),
    "71dB |  91dB peak | -14dB_dolby Reference": Path(r"PEQ_71dB_reference"),
    "70dB |  90dB peak | -15dB_dolby Reference": Path(r"PEQ_70dB_reference"),
    "69dB |  89dB peak | -16dB_dolby Reference": Path(r"PEQ_69dB_reference"),
    "68dB |  88dB peak | -17dB_dolby Reference": Path(r"PEQ_68dB_reference"),
    "67dB |  87dB peak | -18dB_dolby Reference": Path(r"PEQ_67dB_reference"),
    "66dB |  86dB peak | -19dB_dolby Reference": Path(r"PEQ_66dB_reference"),
    "65dB |  85dB peak | -20dB_dolby Reference": Path(r"PEQ_65dB_reference"),
}

# EqualizerAPO config path
EQAPO_CONFIG = Path(r"C:\Program Files\EqualizerAPO\config\config_equal_loudness.txt")


class DragOnlySlider(QSlider):
    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            super().mousePressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        step = 1 if delta > 0 else -1
        new_val = self.value() + step
        self.setValue(max(self.minimum(), min(self.maximum(), new_val)))


# ============================================================
# Biquad calculation based on Audio EQ Cookbook
# ============================================================
def biquad_response(f, fs, filter_type, Fc, Q, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * Fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)

    if filter_type.upper() == "PK":  # Peaking EQ
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif filter_type.upper() == "LS":  # Low shelf
        sqrtA = np.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha
    elif filter_type.upper() == "HS":  # High shelf
        sqrtA = np.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha
    else:  # Default: flat
        return np.ones_like(f)

    # frequency response
    w, h = freqz([b0, b1, b2], [a0, a1, a2], worN=2 * np.pi * f / fs)
    return np.abs(h)


class PEQApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Equal Loudness PEQ Control")
        self.setGeometry(200, 200, 900, 480)

        self.apply_dark_theme()

        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        # Reference selector
        self.ref_label = QLabel("Select Reference Level:")
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(PEQ_FOLDERS.keys())
        self.ref_combo.setStyleSheet("QComboBox { color: black }")
        self.ref_combo.currentIndexChanged.connect(self.on_reference_change)

        control_layout.addWidget(self.ref_label)
        control_layout.addWidget(self.ref_combo)

        # SPL slider
        self.slider_label = QLabel("Select SPL Level:")
        self.slider = DragOnlySlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_change)

        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.slider)

        # Reference box
        ref_box_layout = QHBoxLayout()
        ref_box_label = QLabel("Reference Level (-20dB at 1kHz):")
        self.ref_value_box = QLabel("")
        self.ref_value_box.setStyleSheet(
            "background-color: #400; color: #f88; font-weight: bold; padding: 5px; border-radius: 4px;"
        )
        ref_box_layout.addWidget(ref_box_label)
        ref_box_layout.addWidget(self.ref_value_box)
        control_layout.addLayout(ref_box_layout)

        # Limit checkbox
        self.limit_chk = QCheckBox("Limit Slider to Reference Level")
        self.limit_chk.stateChanged.connect(self.limit_slider_to_ref)
        control_layout.addWidget(self.limit_chk)

        # Auto apply
        self.auto_apply = QCheckBox("Auto Apply")
        self.auto_apply.setChecked(True)
        control_layout.addWidget(self.auto_apply)

        # Show plot checkbox
        self.show_plot_chk = QCheckBox("Show EQ Plot")
        self.show_plot_chk.setChecked(True)
        self.show_plot_chk.stateChanged.connect(self.toggle_plot)
        control_layout.addWidget(self.show_plot_chk)

        # Apply button
        self.apply_btn = QPushButton("Apply PEQ")
        self.apply_btn.setStyleSheet("color: black;")
        self.apply_btn.clicked.connect(self.apply_peq)
        control_layout.addWidget(self.apply_btn)

        control_layout.addStretch()

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setVisible(True)
        self.apply_dark_plot_theme()

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)

        # Initial setup
        self.on_reference_change()
        self.update_plot()

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.Highlight, QColor(100, 150, 255))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)

    def apply_dark_plot_theme(self):
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#2e2e2e')
        self.ax.tick_params(colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(color='gray', linestyle='--')

    # -----------------------------
    # File + slider handling
    # -----------------------------
    def get_available_spls(self, folder: Path):
        spls, max_ref = [], None
        for file in folder.glob("*.txt"):
            name = file.stem.lower().replace(" ", "")
            if "db_to_" in name:
                try:
                    base_spl = int(name.split("db_to_")[0])
                    spls.append(base_spl)
                    after = name.split("to_")[1].replace("db", "")
                    if after.isdigit():
                        max_ref = int(after)
                except ValueError:
                    continue
        return sorted(spls), max_ref

    def on_reference_change(self):
        ref = self.ref_combo.currentText()
        folder = PEQ_FOLDERS[ref]
        spls, ref_max = self.get_available_spls(folder)

        if not spls:
            self.slider.setMinimum(0)
            self.slider.setMaximum(0)
            self.slider_label.setText("No valid PEQ files found!")
            return

        self.slider.setMinimum(min(spls))
        self.slider.setMaximum(max(spls))
        start_val = START_LEVEL if START_LEVEL in spls else min(spls)
        self.slider.setValue(start_val)
        self.current_ref_max = ref_max or max(spls)
        self.ref_value_box.setText(f"{self.current_ref_max} dB Reference")

        self.on_slider_change()
        if self.auto_apply.isChecked():
            self.apply_peq()

    def limit_slider_to_ref(self):
        if self.limit_chk.isChecked() and hasattr(self, "current_ref_max"):
            self.slider.setMaximum(self.current_ref_max)
        elif hasattr(self, "current_ref_max"):
            self.slider.setMaximum(max(self.get_available_spls(PEQ_FOLDERS[self.ref_combo.currentText()])[0]))

    def format_spl_label(self, spl):
        peak = spl + 20
        return f"{spl} dB avg | {peak} dB peak"

    def on_slider_change(self):
        spl = self.slider.value()
        self.slider_label.setText(self.format_spl_label(spl))
        if self.auto_apply.isChecked():
            self.apply_peq()

    # -----------------------------
    # PEQ handling
    # -----------------------------
    def apply_peq(self):
        ref = self.ref_combo.currentText()
        folder = PEQ_FOLDERS[ref]
        spl = self.slider.value()
        target_ref = getattr(self, "current_ref_max", None)

        if not target_ref:
            print("Reference not found.")
            return

        peq_file = folder / f"{spl}dB_to_{target_ref}dB.txt"
        if not peq_file.exists():
            print(f"❌ PEQ file not found for {spl} dB → {target_ref} dB in {ref}")
            return
        try:
            shutil.copy(peq_file, EQAPO_CONFIG)
            print(f"\nApplied {spl} dB PEQ ({ref}) successfully!\n")
            self.update_plot(peq_file)
        except PermissionError:
            print("Permission denied! Run as administrator.")
        except Exception as e:
            print(f"Error: {e}")

    def parse_peq_file(self, path: Path):
        filters = []
        try:
            with open(path, "r") as f:
                for line in f:
                    if "Filter" in line and "ON" in line.upper():
                        parts = line.strip().split()
                        if "Fc" in parts and "Gain" in parts and "Q" in parts:
                            idx_type = parts.index("Filter") + 1
                            filter_type = parts[idx_type]
                            Fc = float(parts[parts.index("Fc") + 1])
                            gain = float(parts[parts.index("Gain") + 1])
                            Q = float(parts[parts.index("Q") + 1])
                            filters.append((filter_type, Fc, Q, gain))
            return filters
        except Exception:
            return []

    def update_plot(self, peq_file=None):
        self.ax.clear()
        self.apply_dark_plot_theme()
        fs = 48000
        f = np.logspace(np.log10(20), np.log10(20000), num=1000)
        H = np.ones_like(f)

        if peq_file and peq_file.exists():
            filters = self.parse_peq_file(peq_file)
            for ft, Fc, Q, gain in filters:
                H *= biquad_response(f, fs, ft, Fc, Q, gain)

        H_db = 20 * np.log10(np.maximum(H, 1e-8))
        self.ax.semilogx(f, H_db, color='cyan', lw=2)
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-15, 15)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Gain (dB)")
        self.ax.set_title("Equalizer Response")
        self.ax.grid(True)
        self.canvas.draw_idle()

    def toggle_plot(self):
        self.canvas.setVisible(self.show_plot_chk.isChecked())


if __name__ == "__main__":
    START_LEVEL = 70
    app = QApplication(sys.argv)
    window = PEQApp()
    window.show()
    sys.exit(app.exec_())
