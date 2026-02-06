import numpy as np
from arms.arm import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa un brazo con distribución Binomial.
        :param n: Número de ensayos (tamaño del lote).
        :param p: Probabilidad de éxito en cada ensayo.
        """
        assert n > 0, "El número de ensayos n debe ser un entero positivo."
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar en el intervalo ."
        
        self.n = int(n)
        self.p = p

    def pull(self):
        """
        Genera una recompensa muestreando de una distribución Binomial.
        Retorna el número de éxitos.
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        El valor esperado de una Binomial es n * p.
        """
        return self.n * self.p

    def __str__(self):
        return f"ArmBinomial(n={self.n}, p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10):
        """
        Genera k brazos con el mismo n pero probabilidades p aleatorias.
        """
        arms =
        for _ in range(k):
            p = np.random.random() # Genera p en = 1 \cdot p = p $$
