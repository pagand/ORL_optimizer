import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from simulator_lstm import CustomEnv
import torch

class SimulatorGUI:
    def __init__(self, env):
        self.env = env
        self.root = tk.Tk()
        self.root.title("Simulation GUI")
        self.root.geometry("800x600")

        # Initialize simulation state
        self.env.reset()
        self.current_step = 0
        self.done = False
        self.distance_history = []

        # Input labels and fields
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=10)

        tk.Label(self.input_frame, text="SPEED").grid(row=0, column=0)
        self.speed_entry = tk.Entry(self.input_frame)
        self.speed_entry.grid(row=0, column=1)

        tk.Label(self.input_frame, text="HEADING").grid(row=0, column=2)
        self.heading_entry = tk.Entry(self.input_frame)
        self.heading_entry.grid(row=0, column=3)

        tk.Label(self.input_frame, text="MODE").grid(row=0, column=4)
        self.mode_entry = tk.Entry(self.input_frame)
        self.mode_entry.grid(row=0, column=5)

        # Step button
        self.step_button = tk.Button(self.root, text="Run Step", command=self.run_step)
        self.step_button.pack(pady=10)

        # Reset button
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="Simulation Status: Running", fg="blue")
        self.status_label.pack()

        # Matplotlib plot
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Distance to Goal")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Distance")
        self.line, = self.ax.plot([], [], label="Distance")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()

    def run_step(self):
        if self.done:
            messagebox.showinfo("Simulation Complete", "Simulation is already finished. Please reset.")
            return

        try:
            # Parse user inputs
            speed = float(self.speed_entry.get()) if self.speed_entry.get() else 0.5
            heading = float(self.heading_entry.get()) if self.heading_entry.get() else 0.5
            mode = int(self.mode_entry.get()) if self.mode_entry.get() else 0

            action = [speed, heading, mode]

            # Run simulation step
            state, reward, done, info = self.env.step(action)
            self.done = done

            # Update plot
            distance = info["state_org"]["distance"]
            self.distance_history.append(distance)
            self.line.set_data(range(len(self.distance_history)), self.distance_history)
            self.ax.set_xlim(0, len(self.distance_history))
            self.ax.set_ylim(0, max(self.distance_history) + 0.1)
            self.canvas.draw()

            # Update status
            self.status_label.config(
                text=f"Step: {self.current_step}, Reward: {reward:.2f}, Distance: {distance:.4f}, Done: {done}",
                fg="green" if not done else "red",
            )

            if done:
                messagebox.showinfo("Simulation Complete", "Goal reached! Simulation ended.")

            self.current_step += 1

        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    def reset_simulation(self):
        self.env.reset()
        self.current_step = 0
        self.done = False
        self.distance_history = []
        self.line.set_data([], [])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
        self.status_label.config(text="Simulation Status: Running", fg="blue")

    def run(self):
        self.root.mainloop()

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
    gui = SimulatorGUI(env)
    gui.run()

