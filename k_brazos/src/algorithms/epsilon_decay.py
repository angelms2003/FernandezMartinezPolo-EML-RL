"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonDecay(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1, epsilon_min:float = 0, decay_lambda: float = 0.001, decay_type: str = "lin"):
        """
        Inicializa el algoritmo epsilon-decaimiento.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :param epsilon_min: Valor mínimo que puede tomar epsilon tras el decaimiento.
        :param decay_lambda: Valor lambda de decaimiento.
        :param decay_type: Tipo de decaimiento a utilizar. Tipos:

            - lin (Decaimiento Lineal, por defecto).

            - exp (Decaimiento Exponencial).

            - inv (Decaimiento Inversamente Proporcional).

        :raises ValueError: Si epsilon no está en [0, 1].
        :raises ValueError: Si decay_type no es "lin", "exp" o "inv".
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."
        assert decay_type in ["lin", "exp", "inv"]

        super().__init__(k)
        self.epsilon = epsilon
        self.t = 0
        self.epsilon_min = epsilon_min
        self.decay_lambda = decay_lambda
        self.decay_type = decay_type

        # Esta variable auxiliar indica si estamos haciendo el recorrido inicial
        # print("="*30)
        # print(f"INICIALIZANDO EPSILON GREEDY CON EPSILON {epsilon}")
        # print("="*30)
        # self.recorrido_inicial = self.epsilon == 0
        self.recorrido_inicial = True

        # Esta variable es un contador que indica por qué brazo vamos al
        # realizar el recorrido inicial en caso de que epsilon sea 0. Al
        # llegar a k, se considera que el recorrido ha terminado
        self.brazo_recorrido_inicial = 0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy con epsilon-decaimiento.

        :return: índice del brazo seleccionado.
        """

        # Se calcula el epsilon del paso temporal actual
        if self.decay_type == "lin":
            epsilon_t = self.epsilon - self.t*self.decay_lambda
        elif self.decay_type == "exp":
            epsilon_t = self.epsilon*np.e**(-self.decay_lambda*self.t)
        elif self.decay_type == "inv":
            epsilon_t = self.epsilon/(1 + self.decay_lambda*self.t)
        
        # Si el epsilon ha bajado demasiado, se establece en el mínimo
        epsilon_t = min(self.epsilon_min, epsilon_t)

        if self.recorrido_inicial:
            # Se está haciendo el recorrido inicial para tener una
            # recompensa promedio inicial para cada uno
            chosen_arm = self.brazo_recorrido_inicial
            self.brazo_recorrido_inicial+=1

            # Si ya se han recorrido todos los brazos, se termina el recorrido inicial
            if self.brazo_recorrido_inicial == self.k:
                self.recorrido_inicial = False

            # print(f"Se está realizando el recorrido inicial. Brazo {chosen_arm}/{self.k}")

        elif np.random.random() < epsilon_t:
            # Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)

            # print(f"Se elige aleatoriamente el brazo {chosen_arm}")
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
            chosen_arm = np.argmax(self.values)

            # print(f"Se elige el mejor brazo {chosen_arm}")
        
        # Aumenta en 1 el paso temporal
        self.t+=1

        return chosen_arm

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()

        self.recorrido_inicial = True
        self.brazo_recorrido_inicial = 0
        self.t = 0