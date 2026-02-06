import numpy as np
from arms.arm import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar en ."
        self.p = p

    def pull(self):
        # Usamos binomial con n=1, que equivale a Bernoulli
        return np.random.binomial(1, self.p)

    def get_expected_value(self) -> float:
        return self.p

    def __str__(self):
        return f"ArmBernoulli(p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int):
        # Genera k brazos con probabilidades p aleatorias uniformes
        return [cls(np.random.random()) for _ in range(k)]
