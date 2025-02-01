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
import pickle
import time


class SimulatorGUI(QMainWindow):
    def __init__(self, env, X_ar):
        super().__init__()

        self.env = env
        self.X_ar = X_ar
        self.ar = -1  # Default: No AR

        self.setWindowTitle("Custom Environment Simulator")
        self.setGeometry(100, 100, 900, 700)

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
        self.ar_input = QLineEdit()
        self.ar_input.setPlaceholderText("AR Steps")

        action_layout.addWidget(QLabel("SPEED:"))
        action_layout.addWidget(self.speed_input)
        action_layout.addWidget(QLabel("HEADING:"))
        action_layout.addWidget(self.heading_input)
        action_layout.addWidget(QLabel("MODE:"))
        action_layout.addWidget(self.mode_input)
        action_layout.addWidget(QLabel("AR Steps:"))
        action_layout.addWidget(self.ar_input)

        main_layout.addLayout(action_layout)

        # Step and Reset buttons
        self.step_button = QPushButton("Run Step")
        self.step_button.clicked.connect(self.run_step)
        main_layout.addWidget(self.step_button)

        self.reset_button = QPushButton("Reset Environment")
        self.reset_button.clicked.connect(self.reset_env)
        main_layout.addWidget(self.reset_button)

        # Status panel
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(self.status_label)

        # Result panel
        self.result_label = QLabel("Result: None")
        self.result_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(self.result_label)

        # Matplotlib Plot
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(2, 3)  # 2 rows, 3 columns

        # Data storage
        self.history = {
            "distance": [], "SFC": [], "SOG": [], "EAT": [], "reward": [], "LAT": [], "LON": [],
            "future": {"distance": [], "SFC": [], "SOG": [], "EAT": [], "reward": [], "LAT": [], "LON": []}
        }

        main_layout.addWidget(self.canvas)
        self.reset_env()

    def reset_env(self):
        """Resets the environment and clears stored data."""
        self.status_label.setText("Status: Resetting the environment")

        i = np.random.randint(0, len(self.X_ar))
        self.state_measured = self.X_ar[i, :, :22]
        self.env.reset(self.state_measured[0])

        for key in self.history:
            if isinstance(self.history[key], dict):  # Clear future predictions
                for subkey in self.history[key]:
                    self.history[key][subkey].clear()
            else:
                self.history[key].clear()

        self.update_plot()
        self.elapsed_time = 0
        self.total_reward = 0
        self.status_label.setText("Status: Reset completed. Ready!")


    def run_step(self):
        """Executes one simulation step and, if applicable, AR steps."""
        try:
            speed = float(self.speed_input.text()) if self.speed_input.text() else -1
            heading = float(self.heading_input.text()) if self.heading_input.text() else -1
            mode = int(self.mode_input.text()) if self.mode_input.text() else -1
            self.ar = int(self.ar_input.text()) if self.ar_input.text() else -1
        except ValueError:
            self.status_label.setText("Status: Invalid input")
            return

        if speed == -1 or heading == -1 or mode == -1:
            self.status_label.setText("Warning: actions cannot be empty, replacing with data")
            if self.elapsed_time >= len(self.state_measured):
                self.status_label.setText("Warning: Repeating last action!")
                seq = -1
            else:
                seq = self.elapsed_time
            action = self.state_measured[seq][:3]
            self.state, reward, done, info = self.env.step(action, self.state_measured[self.elapsed_time][3:])
        else:
            self.status_label.setText("Status: Applying user defined action")
            action = np.array([speed, heading, mode])
            self.state, reward, done, info = self.env.step(action, self.state[3:])

        
        
        # Store current step values
        self.history["distance"].append(info['state_org']['distance'])
        self.history["SFC"].append(info['output_org']['SFC'])
        self.history["SOG"].append(info['output_org']['SOG'])
        self.history["EAT"].append(info['output_org']['EAT'])
        self.history["reward"].append(reward)
        self.history["LAT"].append(info['output_org']['LAT'])
        self.history["LON"].append(info['output_org']['LON'])

        # AR Future Prediction
        horizon = 0
        while self.ar > horizon and self.elapsed_time + horizon + 1 < len(self.state_measured):
            state_ar, reward_ar, done_ar, info_ar = self.env.step(self.state_measured[self.elapsed_time + horizon + 1][:3])
            for key in ["SFC", "SOG", "EAT", "LAT", "LON"]:
                self.history["future"][key].append(info_ar['output_org'][key])
            self.history["future"]["distance"].append(info_ar['state_org']['distance'])
            self.history["future"]["reward"].append(reward_ar)
            horizon += 1

        self.elapsed_time += 1
        self.total_reward += reward
        self.update_plot()

        self.result_label.setText(f"Result: Current reward = {reward:.2f}, Cummulitative reward:{self.total_reward:.2f} , Done = {done}")
        if done:
            self.status_label.setText("Status: Simulation Complete!")
            time.sleep(0.5)
            self.reset_env()
    def update_plot(self):
        """Updates the multi-subplot visualization."""
        self.figure.clear()
        axes = self.figure.subplots(2, 3)

        keys = ["distance", "SFC", "SOG", "EAT", "reward"]
        labels = ["Distance", "SFC", "SOG", "EAT", "Reward"]

        for ax, key, label in zip(axes.flat[:-1], keys, labels):
            ax.clear()
            ax.plot(self.history[key], label=f"{label}", color='b')
            
            if self.history["future"][key]:
                future_x = np.arange(len(self.history[key])-1, len(self.history[key]) + len(self.history["future"][key]))
                future_y = np.array([self.history[key][-1]]+ self.history["future"][key]) # add the actual value in the begining
                std_dev = np.std(future_y) * 0.6  # Assuming 30% variation for uncertainty
                
                ax.fill_between(future_x, future_y - std_dev, future_y + std_dev, color='r', alpha=0.3, label="Future Prediction")
                ax.plot(future_x, future_y, linestyle='dashed', color='r')

            ax.legend()

        # Scatter plot for LAT vs. LON
        axes[1, 2].clear()
        axes[1, 2].scatter(self.history["LAT"], self.history["LON"], label="MAP", c="b", marker=".", s=3)
        # plot the future prediction with empty circles red
        if self.history["future"]["LAT"]:
            axes[1, 2].scatter(self.history["future"]["LAT"], self.history["future"]["LON"], label="Future Prediction",  facecolors='none', edgecolors='red')
        axes[1, 2].set_xlim([49.10, 49.40])
        axes[1, 2].set_ylim([-124, -123.10])
        axes[1, 2].legend()

        self.canvas.draw()

        # clear the future prediction after plot
        for key in self.history["future"]:
            self.history["future"][key].clear() 


if __name__ == "__main__":
    # Create the environment
    model_path = './data/VesselSimulator/lstm_model_ar5.pth'
    scaler_path = './data/VesselSimulator/scaler.pkl'
    start_trip_path = './data/VesselSimulator/start_trip.pkl'
    test_path = './data/VesselSimulator/data_test.pkl'
    with open(test_path, 'rb') as f:
        content = pickle.load(f)
        X1_test_state = content['X1'][0] 
        X2_test_state = content['X2'][0]
    
    X_test_state = np.concatenate((X1_test_state, 
                                   np.concatenate((X2_test_state, np.zeros((5, 2, 23))), axis=1)), axis=0)[:, :, 1:] # ignoring the first state which is done state in dataset

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CustomEnv(
        model_path=model_path,
        start_trip_path=start_trip_path,
        scaler_path=scaler_path,
        device=device
    )

    app = QApplication(sys.argv)
    gui = SimulatorGUI(env, X_ar =X_test_state)
    gui.show()
    sys.exit(app.exec_())
