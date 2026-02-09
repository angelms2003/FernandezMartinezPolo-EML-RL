"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm


class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración.
        """
        assert c >= 0, "El parámetro c debe ser no negativo."

        super().__init__(k)
        self.c = c
        self.t = 0

        # Recorrido inicial para asegurar una estimación inicial
        self.recorrido_inicial = True
        self.brazo_recorrido_inicial = 0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.

        :return: índice del brazo seleccionado.
        """

        if self.recorrido_inicial:
            # Se selecciona cada brazo una vez al inicio
            chosen_arm = self.brazo_recorrido_inicial
            self.brazo_recorrido_inicial += 1

            if self.brazo_recorrido_inicial == self.k:
                self.recorrido_inicial = False

        else:
            # Término de exploración UCB
            exploration = np.sqrt(
                (2 * np.log(self.t + 1)) / self.counts
            )

            # Valores UCB
            ucb_values = self.values + self.c * exploration

            chosen_arm = np.argmax(ucb_values)

        # Incremento del paso temporal
        self.t += 1

        return chosen_arm

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()

        self.t = 0
        self.recorrido_inicial = True
        self.brazo_recorrido_inicial = 0
