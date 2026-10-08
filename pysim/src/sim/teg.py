class TEG:
    def __init__(self, seebeck_coefficient, internal_resistance):
        """
        Parameters
        ----------
        seebeck_coefficient : float
            Effective Seebeck coefficient of the entire TEG [V/K].
        internal_resistance : float
            Electrical internal resistance of the TEG [ohms].
        """
        self.alpha = seebeck_coefficient
        self.Ri = internal_resistance

    def open_circuit_voltage(self, hot_temp, cold_temp):
        """Return open-circuit voltage in volts."""
        delta_T = hot_temp - cold_temp
        return self.alpha * delta_T

    def load_current(self, hot_temp, cold_temp, load_resistance):
        """Return load current in amperes."""
        voc = self.open_circuit_voltage(hot_temp, cold_temp)
        return voc / (self.Ri + load_resistance)

    def electrical_power(self, hot_temp, cold_temp, load_resistance):
        """Return electrical power delivered to the load in watts."""
        current = self.load_current(hot_temp, cold_temp, load_resistance)
        return current**2 * load_resistance

    def maximum_power(self, hot_temp, cold_temp):
        """Return maximum electrical power in watts."""
        voc = self.open_circuit_voltage(hot_temp, cold_temp)
        return voc**2 / (4 * self.Ri)

    def optimal_load_resistance(self):
        """Return load resistance for maximum power."""
        return self.Ri