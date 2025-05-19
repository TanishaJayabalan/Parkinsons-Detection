import tkinter as tk
from pynput import keyboard
import pandas as pd
import numpy as np
import time
import traceback

class FingerTappingAnalyzer:
    def __init__(self, dataset_path, result_label, root):
        self.tap_times = []
        self.test_duration = 10  # seconds
        self.dataset = pd.read_csv(dataset_path)  # Load dataset

        # Handle missing columns with default values
        self.mean_std_tap_time = self.dataset.get('std_tap_time', pd.Series([0])).mean()
        self.mean_cv_tap_time = self.dataset.get('cv_tap_time', pd.Series([0])).mean()
        self.mean_fatigue_index = self.dataset.get('fatigue_index', pd.Series([0])).mean()

        self.result_label = result_label
        self.root = root

    def start_test(self):
        """Start the tapping test with countdown."""
        try:
            self.tap_times = []
            print("Starting countdown...")
            self.update_label("Test starting in:")
            self.countdown(3)
        except Exception as e:
            print("Error in start_test():", e)
            traceback.print_exc()

    def countdown(self, count):
        """Countdown for the test start."""
        try:
            if count > 0:
                print(f"Countdown: {count}...")
                self.update_label(f"{count}...")
                self.root.after(1000, self.countdown, count - 1)
            else:
                print("GO! Start tapping the spacebar!")
                self.update_label("GO! Start tapping the spacebar!")
                self.start_tapping_test()
        except Exception as e:
            print("Error in countdown():", e)
            traceback.print_exc()

    def start_tapping_test(self):
        """Start tapping test and listen for spacebar taps."""
        try:
            self.tap_times = []
            self.listener = keyboard.Listener(on_press=self.on_press)
            print("Starting listener...")
            self.listener.start()

            # Set up a timer using after method to stop the test after the duration
            print(f"Test will stop after {self.test_duration} seconds...")
            self.root.after(self.test_duration * 1000, self.stop_listener)
        except Exception as e:
            print("Error in start_tapping_test():", e)
            traceback.print_exc()

    def on_press(self, key):
        """Record time of spacebar presses."""
        try:
            if key == keyboard.Key.space:
                self.tap_times.append(time.time())
        except Exception as e:
            print("Error in on_press():", e)
            traceback.print_exc()

    def stop_listener(self):
        """Stop the listener after the test duration."""
        try:
            print("Stopping listener...")
            if self.listener:
                self.listener.stop()
            self.analyze_results()
        except Exception as e:
            print("Error in stop_listener():", e)
            traceback.print_exc()

    def analyze_results(self):
        """Analyze tapping data and display results."""
        try:
            if len(self.tap_times) < 2:
                print("Not enough taps recorded.")
                self.update_label("Not enough taps recorded.")
                self.enable_start_button()
                return

            intervals = np.diff(self.tap_times)
            metrics = {
                'mean_tap_time': np.mean(intervals),
                'std_tap_time': np.std(intervals),
                'cv_tap_time': np.std(intervals) / np.mean(intervals),
                'fatigue_index': self._calculate_fatigue_index(intervals),
                'num_taps': len(self.tap_times)
            }

            risk_score = self.calculate_risk(metrics)
            risk_text = self.get_risk_text(risk_score)
            result_message = f"Risk: {risk_text}\nTaps: {metrics['num_taps']}\nMean Tap Time: {metrics['mean_tap_time']:.4f}s"

            print(result_message)
            self.update_label(result_message)
            self.enable_start_button()
        except Exception as e:
            print("Error in analyze_results():", e)
            traceback.print_exc()

    def calculate_risk(self, metrics):
        """Calculate risk based on tapping metrics."""
        try:
            risk_score = 0
            if metrics['std_tap_time'] > self.mean_std_tap_time + 0.02:
                risk_score += 10
            if metrics['cv_tap_time'] > self.mean_cv_tap_time + 0.05:
                risk_score += 15
            if metrics['fatigue_index'] > self.mean_fatigue_index + 0.03:
                risk_score += 25
            return min(risk_score, 100)
        except Exception as e:
            print("Error in calculate_risk():", e)
            traceback.print_exc()

    def _calculate_fatigue_index(self, intervals):
        """Calculate fatigue index."""
        try:
            if len(intervals) < 6:
                return 0
            first = intervals[:len(intervals) // 3]
            last = intervals[-len(intervals) // 3:]
            return (np.mean(last) - np.mean(first)) / np.mean(first)
        except Exception as e:
            print("Error in _calculate_fatigue_index():", e)
            traceback.print_exc()

    def get_risk_text(self, risk_score):
        """Return risk text based on score."""
        try:
            if risk_score <= 30:
                return "Low risk"
            elif risk_score <= 60:
                return "Moderate risk"
            else:
                return "High risk"
        except Exception as e:
            print("Error in get_risk_text():", e)
            traceback.print_exc()

    def update_label(self, text):
        """Update the result label with given text."""
        try:
            self.result_label.config(text=text)
        except Exception as e:
            print("Error in update_label():", e)
            traceback.print_exc()

    def enable_start_button(self):
        """Re-enable the start button after test ends."""
        try:
            print("Re-enabling start button...")
            self.root.after(100, lambda: app.start_button.config(state=tk.NORMAL))
        except Exception as e:
            print("Error in enable_start_button():", e)
            traceback.print_exc()


class App:
    def __init__(self, root, analyzer):
        self.root = root
        self.root.title("Finger Tapping Analyzer")
        self.analyzer = analyzer

        # Create UI components
        self.label = tk.Label(root, text="Finger Tapping Risk Detection", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.start_button = tk.Button(root, text="Start Test", command=self.start_test)
        self.start_button.pack(pady=20)

        self.result_label = tk.Label(root, text="Results will be displayed here.", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        # Set analyzer reference
        self.analyzer.result_label = self.result_label
        self.analyzer.root = self.root

    def start_test(self):
        """Disable the start button and start the test."""
        try:
            print("Test started by button click.")
            self.start_button.config(state=tk.DISABLED)
            self.analyzer.start_test()
        except Exception as e:
            print("Error in start_test():", e)
            traceback.print_exc()

    def run(self):
        """Run the main Tkinter loop."""
        try:
            print("Running Tkinter main loop...")
            self.root.mainloop()
        except Exception as e:
            print("Error in mainloop():", e)
            traceback.print_exc()


# Main program execution
if __name__ == "__main__":
    try:
        root = tk.Tk()
        analyzer = FingerTappingAnalyzer("database.csv", None, root)
        app = App(root, analyzer)
        app.run()
    except Exception as e:
        print("Error during execution:", e)
        traceback.print_exc()
