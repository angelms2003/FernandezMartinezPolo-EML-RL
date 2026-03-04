import numpy as np
import math

from algorithms.algorithm import Algorithm


class Softmax(Algorithm):
    def __init__(self, k: int, tau: float = 1):
        assert 0 < tau, "El parámetro tau debe se mayor que 0."

        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.
        :return: índice del brazo seleccionado.
        """

        numerador = np.exp(self.values / self.tau)
        denominador = np.sum(numerador)
        prob = numerador / denominador

        chosen_arm = np.random.choice(self.k, p=prob)

        return chosen_arm
