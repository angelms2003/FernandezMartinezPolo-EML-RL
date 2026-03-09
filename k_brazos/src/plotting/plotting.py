"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, EpsilonDecay, UCB1, UCB2, Softmax


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, EpsilonDecay):
        label += f" (epsilon={algo.epsilon}, epsilon_min={algo.epsilon_min}, lambda={algo.decay_lambda}, tipo={algo.decay_type})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm], tipo_distribucion:str=None):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param tipo_distribucion: Opcional. String que indica la distribución de brazos usada.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Recompensa Promedio", fontsize=14)
    title = "Recompensa Promedio vs Pasos de Tiempo"
    if tipo_distribucion is not None:
        title += f" ({tipo_distribucion})"
    plt.title(title, fontsize=16)
    plt.legend(title="Algoritmos")
    plt.tight_layout()
    plt.show()


def plot_regret(
    steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], tipo_distribucion:str=None, *args
):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param tipo_distribucion: Opcional. String que indica la distribución de brazos usada.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Rechazo Acumulado", fontsize=14)
    title = "Rechazo Acumulado vs Pasos de Tiempo"
    if tipo_distribucion is not None:
        title += f" ({tipo_distribucion})"
    plt.title(title, fontsize=16)
    plt.legend(title="Algoritmos")
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(
    steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm], tipo_distribucion:str=None
):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param tipo_distribucion: Opcional. String que indica la distribución de brazos usada.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.set_ylim([-5, 105])
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Porcentaje de acción óptima", fontsize=14)
    title = "Porcentaje de acción óptima vs Pasos de Tiempo"
    if tipo_distribucion is not None:
        title += f" ({tipo_distribucion})"
    plt.title(title, fontsize=16)
    plt.legend(title="Algoritmos")
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], 
                        k_arms: int, best_arm_index: int,
                        tipo_distribucion:str=None):
    """
    Visualiza el rendimiento de los brazos mediante una cuadrícula de subplots.
    Muestra la recompensa media obtenida y resalta el brazo óptimo.

    :param arm_stats: Estadísticas (recompensa y conteos) por cada algoritmo.
    :param algorithms: Instancias de los algoritmos para las etiquetas.
    :param k_arms: Cantidad total de brazos.
    :param best_arm_index: Índice (0-indexed) del brazo con mayor recompensa teórica.
    :param tipo_distribucion: Opcional. String que indica la distribución de brazos usada.
    """
    sns.set_context("paper", font_scale=1.1)
    
    n_algos = len(algorithms)
    n_cols = 2
    n_rows = (n_algos + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (algo, stats) in enumerate(zip(algorithms, arm_stats)):
        curr_ax = axes_flat[i]
        
        # Extracción de datos
        avg_rewards = stats.get('mean_rewards', np.zeros(k_arms))
        counts = stats.get('selections', np.zeros(k_arms))
        indices = np.arange(k_arms)
        
        # Definición de estética: verde para el óptimo, gris/azul para el resto
        bar_colors = [sns.color_palette("muted")[2] if j == best_arm_index 
                      else sns.color_palette("muted")[0] for j in indices]

        # Creación del gráfico de barras
        bars = curr_ax.bar(indices, avg_rewards, color=bar_colors, 
                           edgecolor='white', linewidth=1, alpha=0.85)
        
        # Configuración de etiquetas de texto en el eje X
        curr_ax.set_xticks(indices)
        curr_ax.set_xticklabels([f"B{j+1}\n(n={int(counts[j])})" for j in indices], 
                                 fontsize=9)
        
        # Títulos y nombres de ejes
        title = f"Análisis: {get_algorithm_label(algo)}"
        if tipo_distribucion is not None:
            title += f"\n({tipo_distribucion})"
        curr_ax.set_title(title, fontweight='bold')
        curr_ax.set_ylabel("Recompensa Media")
        curr_ax.set_xlabel("Brazo y Frecuencia de Selección")
        curr_ax.yaxis.grid(True, linestyle=':', alpha=0.6)

    # Limpieza de subplots vacíos en caso de número impar
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(pad=3.0)
    plt.show()
