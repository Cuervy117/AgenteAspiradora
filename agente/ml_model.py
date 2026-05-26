"""
=============================================================================
  ml_model.py — El Instinto de Supervivencia (Machine Learning)
=============================================================================
  Imagina que la aspiradora es como un perrito. A veces quiere seguir 
  jugando (explorando), pero su estómago (la batería) ruge. ¿Cómo decide 
  cuándo es el momento exacto de dejar de jugar y volver a casa a comer?

  En lugar de programar reglas aburridas y rígidas como "Vuelve si tu batería
  es menor a 20%", usamos Machine Learning (Un Árbol de Decisión).
  Le enseñamos con 3000 ejemplos simulados qué hacer en diferentes situaciones, 
  y ella "aprende" el concepto de supervivencia por sí sola.
=============================================================================
"""

import os
import pickle
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def get_or_train_model(max_battery=120):
    """
    Carga el cerebro de supervivencia (si ya aprendió antes), o le da 
    clases intensivas si es la primera vez que se ejecuta el programa.
    
    ¿Cómo le enseñamos? (Entrenamiento)
    Creamos 3000 escenarios imaginarios con distintas combinaciones de:
      - Batería que le sobra.
      - Qué tan lejos está su cama (la base).
      - Si ya terminó de explorar o no.
    
    Un "profesor virtual" etiqueta cuál sería la acción correcta en cada
    caso. Luego, el algoritmo analiza esos 3000 ejemplos y extrae las
    "Reglas de Oro" de la supervivencia.
    
    Returns:
        Un modelo inteligente listo para usarse.
    """
    model_path = f"decision_tree_model_{max_battery}.pkl"
    # Si ya fuimos a la escuela antes, simplemente sacamos el diploma del cajón
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # ── Generación de Tareas de Práctica (Datos Sintéticos) ──
    X = [] # Aquí guardamos la situación (ej. "Batería 10%, a 20 pasos")
    y = [] # Aquí guardamos la respuesta correcta (ej. "¡CORRE A CARGAR!")

    for _ in range(3000):
        bat = random.uniform(0.0, 1.0)           # Porcentaje de batería (0 a 1)
        dist = random.randint(0, 30)             # A cuántos pasos está la cama
        exp_done = random.choice([0, 1])         # ¿Terminó de descubrir el mapa?
        pat_done = random.choice([0, 1])         # ¿Terminó de revisar lo que ya limpió?

        # Convertir el porcentaje en "pasos de batería reales"
        bat_raw = bat * max_battery

        # ── El Criterio del Profesor ──
        # La regla de oro: Si la batería APENAS alcanza para llegar a la base
        # (dejando 3 pasitos de sobra por si acaso), ¡abandona todo y regresa!
        if bat_raw <= dist + 3:
            label = 1  # GO_CHARGE (¡Emergencia!)
        else:
            if exp_done == 0:
                label = 0  # EXPLORE (Sigue descubriendo el mundo)
            elif pat_done == 0:
                label = 2  # PATROL (Da una vuelta de vigilancia por donde ya pasaste)
            else:
                label = 3  # GO_FINISH (No hay nada más que hacer, ve a descansar)

        # Anotamos la situación y la respuesta
        X.append([bat, dist, exp_done, pat_done])
        y.append(label)

    # ── El Examen Final (Entrenamiento del Árbol) ──
    # Creamos un "Árbol de Decisiones". Le limitamos la profundidad a 5 ramas
    # (max_depth=5) para que no se aprenda las respuestas de memoria, sino
    # que entienda la lógica de fondo.
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y) # ¡Aquí es donde ocurre la magia del aprendizaje!

    # ── Guardar el cerebro en un frasco (Archivo .pkl) ──
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    return clf


def predict_mode(model, battery_pct, dist_to_base, exp_done, pat_done):
    """
    Le preguntamos al cerebro artificial: "Esta es mi situación actual, ¿qué hago?"
    El modelo baja por sus ramas invisibles en milisegundos y nos da la respuesta.
    """
    # model.predict toma nuestra situación y escupe un número (0, 1, 2, o 3)
    pred = model.predict([[battery_pct, dist_to_base, exp_done, pat_done]])[0]
    
    # Traducimos el número a una palabra que la aspiradora entienda
    modes = {0: "EXPLORE", 1: "GO_CHARGE", 2: "PATROL", 3: "GO_FINISH"}
    return modes[pred]
