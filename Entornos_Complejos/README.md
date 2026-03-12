# Entornos complejos

## Información
- **Alumnos:** Fernández, David; Martínez, Ángel; Polo, Javier
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** FernandezMartinezPolo-EML-RL

## Descripción
Este repositorio presenta un estudio sobre el aprendizaje por refuerzo mediante la evaluación de diferentes algoritmos en dos entornos clásicos del framework Gymnasium: **Taxi-v3** y **MountainCar**. El objetivo es analizar y comparar distintos métodos de aprendizaje por refuerzo en problemas de toma de decisiones secuenciales, considerando tanto entornos discretos como entornos con espacios de estado continuos.

En el entorno **Taxi-v3**, el agente debe aprender a recoger y transportar pasajeros dentro de una cuadrícula, lo que da lugar a un espacio de estados discreto y manejable. Por otro lado, **MountainCar** representa un problema más complejo en el que un coche debe aprender a generar impulso para alcanzar la cima de una montaña, trabajando con un espacio de estados continuo definido por posición y velocidad.

Los métodos tabulares utilizados incluyen las versiones **On-Policy y Off-Policy de Monte Carlo**, así como los algoritmos **SARSA** y **Q-Learning**. Además, se han implementado métodos basados en **aproximación funcional**, concretamente **SARSA semi-gradiente** y **Deep Q-Learning (DQN)**, que permiten abordar entornos donde las representaciones tabulares no son viables.

Los experimentos permiten comparar el comportamiento de estos algoritmos en diferentes tipos de entornos. Los resultados muestran que los **métodos tabulares funcionan adecuadamente en problemas discretos como Taxi-v3**, mientras que **entornos continuos como MountainCar requieren técnicas de aproximación** para poder aprender políticas efectivas. También se analizan las dificultades prácticas encontradas durante el entrenamiento, como la escasez de señal de recompensa en MountainCar, y se exploran técnicas como **reward shaping** para mejorar el aprendizaje.

Finalmente, se discuten las ventajas y limitaciones de cada enfoque, así como posibles líneas de mejora relacionadas con la estabilidad del entrenamiento y la selección de hiperparámetros en métodos basados en redes neuronales.

## Estructura
En el directorio raiz se encuentran los notebooks con los diferentes experimentos realizados, además de un notebook main que contiene enlaces a todos los notebooks. 

Aunque en este caso el código necesario se encuentra en los diferentes notebooks directamente, en el directorio src se encuentra el código python necesario para implementar alguna técnica extra, como el tiling en SARSA semigradiente.

## Instalación y uso

No es necesario instalar nada para ejecutar los notebooks, basta con descargar el repositorio y mantener la estructura. Todos los notebooks tienen un enlace para abrirlos en Google Colab y son reproducibles.

## Tecnologías Utilizadas

Para realizar este trabajo hemos usado Python y notebooks de Jupyter.
