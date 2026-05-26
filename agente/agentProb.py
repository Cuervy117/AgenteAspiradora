from collections import defaultdict
from ml_model import get_or_train_model, predict_mode
from navigation import bfs_path, path_to_actions
from genetic import genetic_algorithm_tsp

class VacuumAgentProb:
    """
    Agente Aspiradora Basado en Probabilidades (Principio de Indiferencia).
    Se adapta a entornos dinámicos calculando riesgos al vuelo sin crear contradicciones lógicas.
    """

    def __init__(self, width, height, start_x, start_y, max_battery=25, start_real_x=0, start_real_y=0):
        self.width = width
        self.height = height
        self.start_real_x = start_real_x
        self.start_real_y = start_real_y
        
        self.x = start_x              
        self.y = start_y              
        self.prev_x = start_x
        self.prev_y = start_y
        self.orientacion = "N"       
        self.base = (start_x, start_y) 

        # Memoria del agente (Mapas)
        self.visited = set()
        self.visited.add((start_x, start_y))
        self.pelusa_positions = set()       
        self.known_carpets = set()          
        # Alfombras inferidas por probabilidad. Se separan de las confirmadas
        # para poder recalcularlas sin dejar falsos positivos pegados.
        self.inferred_carpets = set()
        self.walls = set()
        self.carpet_prob = {}
        self.risking_it = False

        self.max_battery = max_battery
        self.battery = max_battery
        self.path = []               
        self.mode = "EXPLORE"        
        self.known_dirt = set()
        self.visit_count = defaultdict(int)
        
        # ML model
        self.ml_model = get_or_train_model(self.max_battery)
        self.patrol_targets = set()
        self.patrol_route = []

    def _get_valid_adjacents(self, x, y, avoid_risky=False):
        """
        Retorna las celdas cardinales que no son paredes ni alfombras confirmadas.
        Si avoid_risky es True, también evita celdas con riesgo de alfombra.
        """
        adj = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid = []
        for nx, ny in adj:
            if not (0 <= nx + self.start_real_x < self.width and 0 <= ny + self.start_real_y < self.height):
                continue
            if (nx, ny) in self.walls:
                continue
            if (nx, ny) in self.known_carpets:
                continue
            if avoid_risky:
                prob = self.carpet_prob.get((nx, ny), 0.0)
                if prob > 0.0:
                    continue
            valid.append((nx, ny)) 
        return valid

    def perceive(self, percepts):
        """
        Lee el entorno, memoriza posiciones y recalcula al vuelo.
        """
        sensor_frente = percepts.get("Obstaculo_Frente", False)
        sucio = percepts.get("Suciedad", False)
        pelusa = percepts.get("Pelusa", False)

        fx, fy = self.x, self.y
        if self.orientacion == "N": fy += 1
        elif self.orientacion == "S": fy -= 1
        elif self.orientacion == "E": fx += 1
        elif self.orientacion == "W": fx -= 1

        if sensor_frente:
            self.walls.add((fx, fy))
        else:
            if (fx, fy) in self.walls:
                self.walls.remove((fx, fy))

        if sucio:
            self.known_dirt.add((self.x, self.y))
        else:
            if (self.x, self.y) in self.known_dirt:
                self.known_dirt.remove((self.x, self.y))

        if pelusa:
            self.pelusa_positions.add((self.x, self.y))
        else:
            if (self.x, self.y) in self.pelusa_positions:
                self.pelusa_positions.remove((self.x, self.y))

    def _update_probabilities(self):
        """
        Usa el principio de indiferencia para calcular la probabilidad
        de que haya una alfombra en las celdas adyacentes a las pelusas.

        Nota: no cambia la interfaz pública del agente. Solo separa las
        alfombras inferidas de las confirmadas para evitar falsos positivos
        permanentes.
        """
        # Borrar inferencias anteriores antes de recalcular.
        # Así, una celda que antes parecía peligrosa no queda marcada para siempre.
        self.known_carpets.difference_update(self.inferred_carpets)
        self.inferred_carpets.clear()
        self.carpet_prob.clear()
        contributions = defaultdict(list)

        # 1. Identificar celdas seguras por estar en el perímetro de alfombras confirmadas.
        # Las alfombras no pueden estar juntas, ni siquiera en diagonal.
        safe_from_perimeter = set()
        for cx, cy in self.known_carpets:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                sx, sy = cx + dx, cy + dy
                if 0 <= sx + self.start_real_x < self.width and 0 <= sy + self.start_real_y < self.height:
                    safe_from_perimeter.add((sx, sy))

        # 2. Cada pelusa reparte su riesgo uniformemente entre vecinos candidatos.
        for px, py in self.pelusa_positions:
            adj = [(px+1, py), (px-1, py), (px, py+1), (px, py-1)]
            valid_adj = []
            
            for nx, ny in adj:
                if not (0 <= nx + self.start_real_x < self.width and 0 <= ny + self.start_real_y < self.height):
                    continue
                if (nx, ny) in self.visited:
                    continue
                if (nx, ny) in self.walls:
                    continue
                if (nx, ny) in self.known_carpets:
                    continue
                if (nx, ny) in safe_from_perimeter:
                    continue
                valid_adj.append((nx, ny))
            
            if valid_adj:
                prob = 1.0 / len(valid_adj)
                for nx, ny in valid_adj:
                    contributions[(nx, ny)].append(prob)
        
        # 3. Combinar evidencia sin convertir dos señales en certeza absoluta.
        # P(alfombra) = 1 - P(no alfombra por ninguna evidencia)
        for cell, probs in contributions.items():
            p_not_carpet = 1.0
            for p in probs:
                p_not_carpet *= (1.0 - p)
            final_prob = 1.0 - p_not_carpet
            
            self.carpet_prob[cell] = final_prob
            
            # Solo se considera alfombra inferida si la certeza es prácticamente total.
            # Al ser inferida, podrá borrarse y recalcularse en el siguiente ciclo.
            if final_prob >= 0.99:
                self.inferred_carpets.add(cell)
                self.known_carpets.add(cell)

    # Wrappers de módulos externos
    def _bfs_path(self, start, goal, avoid_risky=False):
        def valid_adj(x, y):
            return self._get_valid_adjacents(x, y, avoid_risky)
        return bfs_path(start, goal, valid_adj, self.visited, False)

    def _path_to_actions(self, path_coords, look_only_at_end=False):
        return path_to_actions(path_coords, self.x, self.y, self.orientacion, look_only_at_end)

    def _genetic_algorithm_tsp(self, start, targets):
        def dist_func(n1, n2):
            p = self._bfs_path(n1, n2)
            return len(p) if p else float('inf')
        return genetic_algorithm_tsp(start, targets, dist_func)

    def think_and_act(self):
        """
        Ciclo principal de vida del agente.
        """
        # Actualizar mapa de calor
        self.visit_count[(self.x, self.y)] += 1

        explicacion_total = ""

        self.visited.add((self.x, self.y))
        
        # Eliminar celda actual de los objetivos de patrullaje
        if (self.x, self.y) in self.patrol_targets:
            self.patrol_targets.remove((self.x, self.y))
        if self.patrol_route and (self.x, self.y) in self.patrol_route:
            self.patrol_route.remove((self.x, self.y))

        # 1. ACCIÓN REFLEJA DE LIMPIEZA
        if (self.x, self.y) in self.known_dirt:
            self.battery -= 1
            return "ASPIRAR", (self.x, self.y), explicacion_total + " Basura detectada en mis sensores. Aspirando aquí mismo."
        
        # Recalcular matriz de riesgo de alfombras
        self._update_probabilities()

        if not self.path:
            # IA DE RECARGA (Módulo ML)
            path_coords_to_base = self._bfs_path((self.x, self.y), self.base, avoid_risky=True)
            risking_to_base = False
            if not path_coords_to_base:
                path_coords_to_base = self._bfs_path((self.x, self.y), self.base, avoid_risky=False)
                risking_to_base = True
            
            actions_to_base = self._path_to_actions(path_coords_to_base)
            cost_to_base = len(actions_to_base)

            battery_pct = self.battery / self.max_battery
            exp_done = 1 if self.mode == "GO_FINISH" or self.mode == "PATROL" else 0
            pat_done = 1 if not getattr(self, 'patrol_targets', set()) and not getattr(self, 'patrol_route', []) else 0

            predicted_mode = predict_mode(self.ml_model, battery_pct, cost_to_base, exp_done, pat_done)
            
            if predicted_mode == "GO_CHARGE" and self.mode in ["EXPLORE", "PATROL"] and (self.x, self.y) != self.base:
                self.prev_mode = self.mode
                self.mode = "GO_CHARGE"
                self.path = actions_to_base
                self.risking_it = risking_to_base
                explicacion_total += f"\n[ML] Batería crítica. Volviendo a la base."
                # Permitimos que caiga al bloque normal de ejecución de self.path para que actualice su estado interno.

            if self.mode == "GO_FINISH":
                if (self.x, self.y) == self.base:
                    return "TERMINAR", None, explicacion_total + " Ya estoy en la base."
                else:
                    path_coords = self._bfs_path((self.x, self.y), self.base, avoid_risky=True)
                    self.risking_it = False
                    if not path_coords:
                        path_coords = self._bfs_path((self.x, self.y), self.base, avoid_risky=False)
                        self.risking_it = True
                        
                    self.path = self._path_to_actions(path_coords)
                    if not self.path:
                        return "TERMINAR", None, explicacion_total + " No encontré ruta a la base."

            if self.mode == "GO_CHARGE":
                if (self.x, self.y) == self.base:
                    self.mode = getattr(self, 'prev_mode', 'EXPLORE')
                    self.battery = self.max_battery
                    return "RECARGAR", None, explicacion_total + " Recargando batería."

            if self.mode == "EXPLORE" and self.known_dirt:
                dirt_targets = []
                for d in self.known_dirt:
                    path_coords = self._bfs_path((self.x, self.y), d)
                    if path_coords:
                        dirt_targets.append((d, len(path_coords), path_coords))

                if dirt_targets:
                    dirt_targets.sort(key=lambda x: x[1])
                    target, dist, path_coords = dirt_targets[0]
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += f"\n[MEMORIA] Recordé basura en {target}. Voy para allá.\n"
                    
            if self.mode == "EXPLORE" and not self.path:
                frontier = set()
                for vx, vy in self.visited:
                    for nx, ny in self._get_valid_adjacents(vx, vy):
                        if (nx, ny) not in self.visited:
                            frontier.add((nx, ny))

                safe_frontiers = []
                unknown_frontiers = []
                risky_frontiers = []

                for fx, fy in list(frontier):
                    if (fx, fy) in self.known_carpets:
                        frontier.remove((fx, fy))
                        continue

                    path_coords = self._bfs_path((self.x, self.y), (fx, fy))
                    if not path_coords:
                        continue # Inalcanzable
                    
                    dist = len(path_coords)
                    prob = self.carpet_prob.get((fx, fy), 0.0)
                    
                    is_near_pelusa = any(
                        (fx + dx, fy + dy) in self.pelusa_positions
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )

                    if prob < 0.1 and not is_near_pelusa:
                        safe_frontiers.append(((fx, fy), dist, "Riesgo probabilístico muy bajo."))
                    elif prob < 0.5:
                        unknown_frontiers.append(((fx, fy), dist))
                    else:
                        risky_frontiers.append(((fx, fy), dist))

                if safe_frontiers:
                    safe_frontiers.sort(key=lambda x: x[1])
                    path_found = False
                    for target, dist, expl in safe_frontiers:
                        path_coords = self._bfs_path((self.x, self.y), target, avoid_risky=True)
                        if path_coords:
                            self.path = self._path_to_actions(path_coords)
                            self.risking_it = False
                            explicacion_total += f"\n(Probabilidad) [EXPLORE] {target} es segura.\n{expl}"
                            path_found = True
                            break
                    if not path_found:
                        safe_frontiers = [] # Fallback a arriesgar
                        
                if not self.path and unknown_frontiers:
                    unknown_frontiers.sort(key=lambda x: x[1])
                    target, dist = unknown_frontiers[0]
                    path_coords = self._bfs_path((self.x, self.y), target)
                    self.path = self._path_to_actions(path_coords, look_only_at_end=False)
                    self.risking_it = True
                    explicacion_total += f"\nNo sé qué hay en {target} ({self.carpet_prob.get(target, 0.0)*100:.1f}% alfombra). Me acerco a investigar.\n"
                elif not self.path and risky_frontiers:
                    risky_frontiers.sort(key=lambda x: x[1])
                    target, dist = risky_frontiers[0]
                    path_coords = self._bfs_path((self.x, self.y), target)
                    self.path = self._path_to_actions(path_coords, look_only_at_end=False)
                    self.risking_it = True
                    explicacion_total += f"\n[RIESGO ALTO] Explorando {target} ({self.carpet_prob.get(target, 0.0)*100:.1f}% alfombra) como último recurso.\n"
                elif not self.path:
                    self.mode = "GO_FINISH"
                    path_coords = self._bfs_path((self.x, self.y), self.base)
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += "Reconocimiento completado. Regreso a la base.\n"

            if self.mode == "PATROL":
                if self.patrol_targets:
                    explicacion_total += "Creando la ruta para vigilar...\n"
                    self.patrol_route = self._genetic_algorithm_tsp((self.x, self.y), list(self.patrol_targets))
                    self.patrol_targets = set()

                if hasattr(self, 'patrol_route') and self.patrol_route:
                    target = self.patrol_route.pop(0)
                    self.current_target = target
                    path_coords = self._bfs_path((self.x, self.y), target, avoid_risky=True)
                    if not path_coords: # Fallback sin seguridad si es inalcanzable
                        path_coords = self._bfs_path((self.x, self.y), target, avoid_risky=False)
                    self.path = self._path_to_actions(path_coords)
                else:
                    self.mode = "GO_FINISH"
                    path_coords = self._bfs_path((self.x, self.y), self.base)
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += "Termino el patrullaje. Regreso a la base.\n"

                    if not self.path:
                        if (self.x, self.y) == self.base:
                            return "TERMINAR", None, explicacion_total + " Ya estoy en la base."


        if self.path:
            next_action = self.path.pop(0)
            
            if isinstance(next_action, tuple) and next_action[0] == "MOVER":
                next_pos = next_action[1]
                
                # Caso especial. Generación espontanea de obstáculos.
                if next_pos in self.walls or next_pos in self.known_carpets:
                    self.path = [] 
                    obs_name = "un muro" if next_pos in self.walls else "una alfombra"
                    explicacion_total += f"\n¡El usuario puso {obs_name} en {next_pos}! Calculando ruta nueva."
                    return "ESPERAR", None, explicacion_total + " Recalculando..."



                self.prev_x, self.prev_y = self.x, self.y
                self.x, self.y = next_pos
                self.battery -= 1
                return "MOVER", next_pos, explicacion_total + f" [RUTA] Moviendome hacia {next_pos}"

            elif next_action == "ROTAR_IZQ":
                dirs = ["N", "E", "S", "W"]
                self.orientacion = dirs[(dirs.index(self.orientacion) - 1) % 4]
                self.battery -= 1
                return "ROTAR_IZQ", None, explicacion_total + f" [RUTA] Rotando a la izquierda."
                
            elif next_action == "ROTAR_DER":
                dirs = ["N", "E", "S", "W"]
                self.orientacion = dirs[(dirs.index(self.orientacion) + 1) % 4]
                self.battery -= 1
                return "ROTAR_DER", None, explicacion_total + f" [RUTA] Rotando a la derecha."
            
            elif next_action == "ASPIRAR":
                return "ASPIRAR", None, explicacion_total
        
        return "ESPERAR", None, explicacion_total + " Pensando..."

    def get_planned_route(self):
        """
        Le pasa su plan a PyGame para que dibuje la línea azul claro.
        """
        route = []
        for action in self.path:
            if isinstance(action, tuple) and action[0] == "MOVER":
                route.append(action[1])
        return route
