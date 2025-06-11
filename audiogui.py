import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QCheckBox, QPushButton, QFileDialog
from fft_filters import peak_picking, harmonic_masking, smoothing

SAMPLE_RATE = 44100
FFT_SIZE = 1024
HOP_SIZE = 512
window = np.hanning(FFT_SIZE)

import csv

class FFTApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time FFT Visualizer")

        self.layout = QVBoxLayout()
        self.label = QLabel("Select input device:")
        self.combo = QComboBox()
        self.plot_widget = pg.PlotWidget()

        self.peak_checkbox = QCheckBox("Peak Picking")
        self.mask_checkbox = QCheckBox("Mask Harmonics")
        self.smooth_checkbox = QCheckBox("Smoothing")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.peak_checkbox)
        self.layout.addWidget(self.mask_checkbox)
        self.layout.addWidget(self.smooth_checkbox)
        self.save_button = QPushButton("Save FFT to CSV")
        self.save_button.clicked.connect(self.capture_fft_snapshot)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        self.devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        for i, d in enumerate(self.devices):
            self.combo.addItem(f"{i}: {d['name']}")

        self.combo.currentIndexChanged.connect(self.start_stream)

        self.note_names, self.note_positions = self.generate_note_labels()
        self.fft_bars = pg.BarGraphItem(x=self.note_positions, height=[0]*len(self.note_positions), width=1, brush='y')
        self.plot_widget.addItem(self.fft_bars)
        self.max_magnitude = 1.0
        self.plot_widget.setYRange(0, self.max_magnitude)
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setXRange(0, len(self.note_positions))
        self.note_labels = self.generate_note_labels()
        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([[ (pos, name) for pos, name in zip(self.note_positions, self.note_names) if pos > 0 ]])

        self.buffer = np.zeros(FFT_SIZE)
        self.fft_data = np.zeros(FFT_SIZE // 2 + 1)
        self.stream = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)

    def start_stream(self, index):
        if self.stream:
            self.stream.close()

        device_index = self.devices[index]['index']
        sd.default.device = (device_index, None)

        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            callback=self.audio_callback
        )
        self.last_freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        self.last_mag_raw = np.zeros_like(self.last_freqs)
        self.last_mag_filtered = np.zeros_like(self.last_freqs)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        self.buffer = np.roll(self.buffer, -HOP_SIZE)
        self.buffer[-HOP_SIZE:] = indata[:, 0]
        windowed = self.buffer * window
        fft = np.abs(np.fft.rfft(windowed))
        self.fft_data = fft

    def generate_note_labels(self):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_names = []
        positions = []
        for midi in range(21, 109):  # A0 to C8
            freq = 440 * 2 ** ((midi - 69) / 12)
            if 27.5 <= freq <= 4186:
                note = notes[midi % 12] + str(midi // 12 - 1)
                note_names.append(note)
                positions.append(midi - 21)
        return note_names, positions

    def update_plot(self):
        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        mag = self.fft_data.copy()

        self.last_freqs = freqs
        self.last_mag_raw = mag.copy()

        if self.peak_checkbox.isChecked():
            mag = peak_picking(mag)
        if self.mask_checkbox.isChecked():
            mag = harmonic_masking(mag, SAMPLE_RATE, FFT_SIZE)
        if self.smooth_checkbox.isChecked():
            mag = smoothing(mag)

        self.last_mag_filtered = mag.copy()

        peak = np.max(mag)
        if peak > self.max_magnitude:
            self.max_magnitude = peak
            self.plot_widget.setYRange(0, self.max_magnitude)

                        # Convert frequencies to note bin indexes
        note_bins = []
        for i, f in enumerate(freqs):
            if f <= 0:
                continue
            try:
                midi = int(round(69 + 12 * np.log2(f / 440)))
                index = midi - 21
                if 0 <= index < len(self.note_positions):
                    note_bins.append((index, mag[i]))
            except ValueError:
                continue


        bar_heights = [0] * len(self.note_positions)
        for i, h in note_bins:
            if i < len(bar_heights):
                bar_heights[i] = max(bar_heights[i], h)  # max magnitude per note bin

        self.fft_bars.setOpts(x=self.note_positions, height=bar_heights, width=1, brush='y')

    def capture_fft_snapshot(self):
        self.snapshot_freqs = self.last_freqs.copy()
        self.snapshot_raw = self.last_mag_raw.copy()
        self.snapshot_filtered = self.last_mag_filtered.copy()
        self.save_fft_to_csv()

    def save_fft_to_csv(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save FFT Data", "fft_data.csv", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Frequency (Hz)", "Raw Magnitude", "Filtered Magnitude"])
                for freq, raw, filt in zip(self.snapshot_freqs, self.snapshot_raw, self.snapshot_filtered):
                    writer.writerow([f"{freq:.2f}", f"{raw:.6f}", f"{filt:.6f}"])

