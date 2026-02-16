import numpy as np
from .algorithm import Algorithm


class Softmax(Algorithm):
    def __init__(self, k, tau):
        super().__init__(k)
        self.tau = tau

    def select_arm(self):
        # Obtener los valores Q actuales
        q_values = self.values

        # Estabilidad numérica: restar el máximo valor Q antes de exponenciar
        # Esto previene overflow en np.exp()
        # z = (Q(a) / tau) - max(Q(a) / tau)
        tau_scaled_q = q_values / self.tau
        max_q = np.max(tau_scaled_q)
        shifted_q = tau_scaled_q - max_q

        # Calcular exponenciales
        exp_values = np.exp(shifted_q)

        # Calcular probabilidades (distribución de Boltzmann)
        probabilities = exp_values / np.sum(exp_values)

        # Selección estocástica basada en las probabilidades calculadas
        chosen_arm = np.random.choice(range(self.k), p=probabilities)
        return chosen_arm

    def __str__(self):
        return f"Softmax (tau={self.tau})"
