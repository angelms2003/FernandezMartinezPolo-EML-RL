"""
Module: algorithms/ucb2.py
Description: Implementación del algoritmo UCB2 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2026/02/09

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):

    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste de exploración (0 < alpha <= 1).
        """
        assert 0 < alpha <= 1, "El parámetro alpha debe estar en (0,1]."

        super().__init__(k)
        self.alpha = alpha
        self.t = 0

        # Para cada brazo, almacenar hasta qué instante debemos seguir seleccionándolo
        self.next_update = np.zeros(k, dtype=int)
        self.current_arm = None
        self.arm_count_in_epoch = 0

        # Recorrido inicial para asegurar una selección inicial
        self.recorrido_inicial = True
        self.brazo_recorrido_inicial = 0

    def tau(self, r: int) -> int:
        """
        Función UCB2: duración del epoch de un brazo.
        """
        return int(np.ceil((1 + self.alpha) ** r))

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB2.

        :return: índice del brazo seleccionado.
        """

        # Recorrido inicial: seleccionar cada brazo al menos una vez
        if self.recorrido_inicial:
            chosen_arm = self.brazo_recorrido_inicial
            self.brazo_recorrido_inicial += 1
            if self.brazo_recorrido_inicial == self.k:
                self.recorrido_inicial = False

            self.current_arm = chosen_arm
            self.arm_count_in_epoch = 1
            return chosen_arm

        # Verificar si debemos cambiar de brazo (fin del epoch actual)
        if self.arm_count_in_epoch >= self.next_update[self.current_arm]:
            # Calcular UCB para decidir nuevo brazo
            ucb_values = self.values + np.sqrt(
                (1 + self.alpha) * np.log(np.e * self.t / self.counts) / (2 * self.counts)
            )

            self.current_arm = int(np.argmax(ucb_values))
            # Duración del siguiente epoch
            r = int(np.ceil(np.log(self.t + 1) / np.log(1 + self.alpha)))
            self.next_update[self.current_arm] = self.t + self.tau(r)
            self.arm_count_in_epoch = 1

        else:
            self.arm_count_in_epoch += 1

        self.t += 1
        return self.current_arm

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()
        self.t = 0
        self.next_update = np.zeros(self.k, dtype=int)
        self.current_arm = None
        self.arm_count_in_epoch = 0
        self.recorrido_inicial = True
        self.brazo_recorrido_inicial = 0
