import os
import pickle
import random
from sklearn.tree import DecisionTreeClassifier

def get_or_train_model(max_battery=120):
    """
    Entrenamiento de modelo.
    Args: 
        max_battery (int): Capacidad máxima de la batería.

    Returns: 
        Un modelo inteligente listo para usarse.
    """
    model_path = f"decision_tree_model_{max_battery}.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # Training dataset
    X = [] # Situación: Porcentaje de batería, distancia a la base, si terminó de explorar, si terminó de revisar lo que ya limpió.
    y = [] # Respuesta correcta: EXPLORE, GO_CHARGE, PATROL, GO_FINISH.

    for _ in range(3000):
        bat = random.uniform(0.0, 1.0)           # Porcentaje de batería (0 a 1)
        dist = random.randint(0, 30)             # A cuántos pasos está la base
        exp_done = random.choice([0, 1])         # ¿Terminó de descubrir el mapa?
        pat_done = random.choice([0, 1])         # ¿Terminó de revisar lo que ya limpió?

        # Convertimos el porcentaje en pasos.
        bat_raw = bat * max_battery

        # Si la bateria es justa para regresar a base
        if bat_raw <= dist + 3:
            label = 1  # GO_CHARGE
        else:
            if exp_done == 0:
                label = 0  # EXPLORE
            elif pat_done == 0:
                label = 2  # PATROL
            else:
                label = 3  # GO_FINISH

        # Anotamos la situación y la respuesta
        X.append([bat, dist, exp_done, pat_done])
        y.append(label)

    # Creación del arbol de deciciones.
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y) 

    # Guardamos el modelo.
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    return clf


def predict_mode(model, battery_pct, dist_to_base, exp_done, pat_done):
    """
    Args: 
        model (DecisionTreeClassifier): Modelo entrenado.
        battery_pct (float): Porcentaje de batería.
        dist_to_base (int): Distancia a la base.
        exp_done (int): Si terminó de explorar.
        pat_done (int): Si terminó de revisar lo que ya limpió.

    Returns: 
        str: Acción a realizar.
    """
    pred = model.predict([[battery_pct, dist_to_base, exp_done, pat_done]])[0]
    modes = {0: "EXPLORE", 1: "GO_CHARGE", 2: "PATROL", 3: "GO_FINISH"}
    return modes[pred]
