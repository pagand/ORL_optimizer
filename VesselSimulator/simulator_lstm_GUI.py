import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from simulator_lstm import CustomEnv
import torch

class SimulatorGUI(QMainWindow):
    def __init__(self, env):
        super().__init__()

        self.env = env
        self.state = self.env.reset()

        self.setWindowTitle("Custom Environment Simulator")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Action input fields
        action_layout = QHBoxLayout()
        self.speed_input = QLineEdit()
        self.speed_input.setPlaceholderText("Speed")
        self.heading_input = QLineEdit()
        self.heading_input.setPlaceholderText("Heading")
        self.mode_input = QLineEdit()
        self.mode_input.setPlaceholderText("Mode")

        action_layout.addWidget(QLabel("SPEED:"))
        action_layout.addWidget(self.speed_input)
        action_layout.addWidget(QLabel("HEADING:"))
        action_layout.addWidget(self.heading_input)
        action_layout.addWidget(QLabel("MODE:"))
        action_layout.addWidget(self.mode_input)

        main_layout.addLayout(action_layout)

        # Step button
        self.step_button = QPushButton("Run Step")
        self.step_button.clicked.connect(self.run_step)
        main_layout.addWidget(self.step_button)

        # Status panel
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(self.status_label)

        # Matplotlib plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Simulation Progress")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Distance to Goal")
        self.distance_data = []

        main_layout.addWidget(self.canvas)

    def run_step(self):
        # Get actions from input fields
        try:
            speed = float(self.speed_input.text()) if self.speed_input.text() else 0.5
            heading = float(self.heading_input.text()) if self.heading_input.text() else 0.5
            mode = int(self.mode_input.text()) if self.mode_input.text() else 0
            action = np.array([speed, heading, mode])
        except ValueError:
            self.status_label.setText("Status: Invalid action input")
            return

        # Run simulation step
        self.state, reward, done, info = self.env.step(action)
        distance = info['state_org']['distance']

        # Update plot
        self.distance_data.append(distance)
        self.ax.clear()
        self.ax.plot(self.distance_data, label="Distance to Goal")
        self.ax.set_title("Simulation Progress")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Distance to Goal")
        self.ax.legend()
        self.canvas.draw()

        # Update status
        self.status_label.setText(f"Status: Reward = {reward:.2f}, Done = {done}")

        # Reset if simulation ends
        if done:
            self.status_label.setText("Status: Simulation Complete! Resetting...")
            self.state = self.env.reset()
            self.distance_data.clear()
            self.ax.clear()
            self.ax.set_title("Simulation Progress")
            self.ax.set_xlabel("Step")
            self.ax.set_ylabel("Distance to Goal")
            self.canvas.draw()

if __name__ == "__main__":
    # Create the environment
    model_path = './data/VesselSimulator/lstm_model.pth'
    scaler_path = './data/VesselSimulator/scaler.pkl'
    start_trip_path = './data/VesselSimulator/start_trip.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CustomEnv(
        model_path=model_path,
        start_trip_path=start_trip_path,
        scaler_path=scaler_path,
        device=device
    )

    # Run the application
    app = QApplication(sys.argv)
    gui = SimulatorGUI(env)
    gui.show()
    sys.exit(app.exec_())
