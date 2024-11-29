import numpy as np


class Evaluation:
    def __init__(self, actual_values, predicted_values, n, p):
        self.actual_values = actual_values
        self.predicted_values = predicted_values
        self.n = n
        self.p = p

    def mean_absolute_error(self):
        return np.mean(np.abs(self.actual_values - self.predicted_values))

    def root_mean_squared_error(self):
        return np.sqrt(np.mean((self.actual_values - self.predicted_values) ** 2))

    def mean_absolute_percentage_error(self):
        return np.mean(np.abs((self.actual_values - self.predicted_values) / self.actual_values)) * 100

    def mean_error(self):
        return np.mean(self.actual_values - self.predicted_values)

    def symmetric_mean_absolute_percentage_error(self):
        actual = np.array(self.actual_values)
        predicted = np.array(self.predicted_values)

        return np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100

    def r_squared(self):
        ss_res = np.sum((self.actual_values - self.predicted_values) ** 2)
        ss_tot = np.sum((self.actual_values - np.mean(self.actual_values)) ** 2)
        print(f"SS_Res: {ss_res}, SS_Tot: {ss_tot}")  # Debugging line
        return 1 - (ss_res / ss_tot)

    def adjusted_r_squared(self):
        r2 = self.r_squared()
        return 1 - (1 - r2) * (self.n - 1) / (self.n - self.p - 1)
