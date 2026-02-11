"""
Module: algorithms/ucb2.py
Description: Implementación del algoritmo UCB2 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2026/02/11

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):

    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2 con gestión interna de tiempo.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste de exploración (0 < alpha < 1).
        """
        assert 0 < alpha < 1, "El parámetro alpha debe estar entre 0 y 1."

        super().__init__(k)
        self.alpha = alpha
        self.t = 0  # Contador de tiempo interno
        
        # kas[i] almacena el número de épocas que se ha jugado el brazo i
        self.kas = np.zeros(k, dtype=int)
        
        # Gestión de ráfagas (epochs)
        self.current_arm = None
        self.arm_count_in_epoch = 0
        self.total_plays_needed = 0

    def tau(self, r: int) -> float:
        """
        Función que define el límite superior del número de selecciones para una fase.
        """
        return (1 + self.alpha)**r

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB2.

        :return: Índice del brazo seleccionado.
        """
        
        # 1. Si estamos en medio de una ráfaga, seguimos con el mismo brazo
        if self.current_arm is not None and self.arm_count_in_epoch < self.total_plays_needed:
            self.arm_count_in_epoch += 1
            self.t += 1
            return self.current_arm

        # 2. Si la ráfaga terminó (o es el inicio), seleccionamos nuevo brazo
        # Primero: asegurar recorrido inicial (un epoch por brazo)
        for i in range(self.k):
            if self.counts[i] == 0:
                self.current_arm = i
                self._prepare_new_epoch()
                self.t += 1
                return self.current_arm

        # Segundo: Calcular valores UCB según la fórmula de la segunda versión
        # n_a = ceil(tau(k_a))
        n_a = np.ceil(self.tau(self.kas))
        
        # Término de exploración utilizando el self.t interno
        # Se usa (self.t + 1) para evitar log(0) en el primer paso tras inicialización
        exploration = np.sqrt(
            ((1 + self.alpha) * np.log(np.e * self.t/ n_a)) / (2 * n_a)
        )
        
        ucb_values = self.values + exploration
        
        # Selección del brazo con el índice UCB más alto
        self.current_arm = int(np.argmax(ucb_values))
        self._prepare_new_epoch()
        
        self.t += 1
        return self.current_arm

    def _prepare_new_epoch(self):
        """
        Calcula la duración de la ráfaga para el brazo actual e incrementa su contador de fases.
        """
        r = self.kas[self.current_arm]
        # Duración: ceil(tau(r+1)) - ceil(tau(r))
        self.total_plays_needed = int(np.ceil(self.tau(r + 1)) - np.ceil(self.tau(r)))
        self.arm_count_in_epoch = 1
        self.kas[self.current_arm] += 1

    def reset(self):
        """
        Reinicia completamente el estado del algoritmo.
        """
        super().reset()
        self.t = 0
        self.kas = np.zeros(self.k, dtype=int)
        self.current_arm = None
        self.arm_count_in_epoch = 0
        self.total_plays_needed = 0