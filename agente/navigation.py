from collections import deque

def bfs_path(start, goal, valid_adjacents_func, visited_set, allow_unknown=False):
    """
    Encuentra la ruta más corta de 'start' a 'goal' utilizando Búsqueda en Anchura (BFS).
    Evita paredes y alfombras conocidas.

    Args:
        start: Tupla (x, y) de inicio.
        goal: Tupla (x, y) objetivo.
        valid_adjacents_func: Función que recibe (x, y) y retorna los vecinos transitables.
        visited_set: Conjunto de casillas que el agente ya ha visitado de forma segura.
        allow_unknown: Si es True, permite planear rutas cruzando territorio sin explorar 
                       (niebla de guerra) siempre y cuando no esté marcado como peligroso.
    
    Returns:
        Lista de coordenadas que componen la ruta.
    """
    queue   = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path[1:]

        for nxt in valid_adjacents_func(*node):
            # Si allow_unknown es False, solo se permite viajar a través de
            # casillas ya visitadas o hacia la meta final.
            can_visit = (nxt in visited_set or nxt == goal) if not allow_unknown else True
            if nxt not in visited and can_visit:
                visited.add(nxt)
                queue.append(path + [nxt])

    return []


def path_to_actions(path_coords, curr_x, curr_y, curr_dir, look_only_at_end=False):
    """
    Convierte una lista de coordenadas (x,y) en un conjunto de instrucciones
    físicas para el agente: ROTAR_IZQ, ROTAR_DER, MOVER.
    """
    actions  = []
    dirs     = ["N", "E", "S", "W"]

    for i, (nx, ny) in enumerate(path_coords):
        if   nx == curr_x and ny == curr_y + 1: target_dir = "N"
        elif nx == curr_x and ny == curr_y - 1: target_dir = "S"
        elif nx == curr_x + 1 and ny == curr_y: target_dir = "E"
        elif nx == curr_x - 1 and ny == curr_y: target_dir = "W"
        else: continue

        while curr_dir != target_dir:
            idx   = dirs.index(curr_dir)
            t_idx = dirs.index(target_dir)
            if   (idx + 1) % 4 == t_idx:
                actions.append("ROTAR_DER"); curr_dir = dirs[(idx+1)%4]
            elif (idx - 1) % 4 == t_idx:
                actions.append("ROTAR_IZQ"); curr_dir = dirs[(idx-1)%4]
            elif (idx + 2) % 4 == t_idx:
                actions.append("ROTAR_DER"); curr_dir = dirs[(idx+1)%4]

        is_last = (i == len(path_coords) - 1)
        if not (is_last and look_only_at_end):
            actions.append(("MOVER", (nx, ny)))
            curr_x, curr_y = nx, ny

    return actions
