import random

def genetic_algorithm_tsp(start, targets, dist_func):
    """
    Resuelve el Problema del Viajante de Comercio (TSP) usando un algoritmo genético.
    
    Args:
        start: Tupla con la coordenada inicial.
        targets: Lista de tuplas con las coordenadas objetivo.
        dist_func: Función que recibe (nodo1, nodo2) y retorna la distancia entre ellos.
    
    Returns:
        Una lista de coordenadas que representa la ruta óptima.
    """
    if not targets: return []
    if len(targets) == 1: return targets

    def route_distance(route):
        d = dist_func(start, route[0])
        for i in range(len(route)-1):
            d += dist_func(route[i], route[i+1])
        return d

    pop_size  = max(10, min(50, len(targets)*2))
    gens      = 40

    # Semilla greedy
    nn_route, unvisited, curr = [], set(targets), start
    while unvisited:
        closest = min(unvisited, key=lambda x: dist_func(curr, x))
        nn_route.append(closest)
        unvisited.remove(closest)
        curr = closest

    population = [nn_route] + [random.sample(targets, len(targets)) for _ in range(pop_size-1)]

    for _ in range(gens):
        population.sort(key=route_distance)
        elite    = population[:pop_size//2]
        next_pop = elite.copy()
        while len(next_pop) < pop_size:
            p1, p2    = random.sample(elite, 2)
            s, e      = sorted(random.sample(range(len(targets)), 2))
            child     = [None]*len(targets)
            child[s:e] = p1[s:e]
            p2_idx = 0
            for i in range(len(targets)):
                if child[i] is None:
                    while p2[p2_idx] in child: p2_idx += 1
                    child[i] = p2[p2_idx]
            if random.random() < 0.2: # Mutación
                m1, m2 = random.sample(range(len(targets)), 2)
                child[m1], child[m2] = child[m2], child[m1]
            next_pop.append(child)
        population = next_pop

    population.sort(key=route_distance)
    return population[0]
