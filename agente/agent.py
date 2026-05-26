from logic import KnowledgeBase, Literal
from sympy import symbols, Equivalent, Or, Not, And
from collections import defaultdict
from ml_model import get_or_train_model, predict_mode
from navigation import bfs_path, path_to_actions
from genetic import genetic_algorithm_tsp

class VacuumAgent:
    """
    Clase que representa el agente.
    """

    def __init__(self, width, height, start_x, start_y, max_battery=25):
        self.width = width
        self.height = height
        
        self.x = start_x              
        self.y = start_y              
        self.prev_x = start_x
        self.prev_y = start_y
        self.orientacion = "N"       
        self.base = (start_x, start_y) 

        # Memoria del agente
        self.visited = set()
        self.visited.add((start_x, start_y))
        self.pelusa_positions = set()       
        self.known_carpets = set()          
        self._pelusa_rules_added = set()    
        self.kb = KnowledgeBase()

        self.max_battery = max_battery
        self.battery = max_battery

        # Tareas del agente
        self.mode = "EXPLORE"         
        self.path = []                
        self.patrol_targets = set()   
        self.patrol_route = []        
        self.current_target = None   
        self.known_dirt = set()       
        self.dist_cache = {}          
        self.visit_count = defaultdict(int) 

        # ML. Inicialización del modelo.
        self.ml_model = get_or_train_model(self.max_battery)

        # Inicialización de reglas (KB)
        self._init_rules()

    # Percepción inicial
    def _init_rules(self):
        """
        Al inicio, el agente sabe que no hay un muro ni una alfombra en su posición inicial.
        """
        self.kb.add_fact(Literal(f"O_{self.x}_{self.y}").negate())
        self.kb.add_fact(Literal(f"C_{self.x}_{self.y}").negate())

    def _get_valid_adjacents(self, x, y):
        """
        Función auxiliar. Identifica las casillas adyacentes válidas (no hay muro).
        Args:
            x: Coordenada x
            y: Coordenada y
        Returns:
            Lista de coordenadas adyacentes válidas
        """
        adj = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid = []
        for nx, ny in adj:
            if (nx, ny) in self.known_carpets:
                continue  # Alfombra confirmada por inferencia, intransitable
            obs_sym = Literal(f"O_{nx}_{ny}").to_sympy()
            if obs_sym not in self.kb.clauses:
                valid.append((nx, ny)) 
        return valid

    # Sensores. Percepciones

    def perceive(self, percepts):
        """
        Recibe las percepciones del entorno y actualiza la base de conocimiento.
        Args:
            percepts: Diccionario con las percepciones
        Returns:
            None
        """
        sensor_frente = percepts.get("Obstaculo_Frente", False)
        sucio = percepts.get("Suciedad", False)
        pelusa = percepts.get("Pelusa", False)

        # Posición matemática
        fx, fy = self.x, self.y
        if self.orientacion == "N": fy += 1
        elif self.orientacion == "S": fy -= 1
        elif self.orientacion == "E": fx += 1
        elif self.orientacion == "W": fx -= 1

        # Si el sensor láser detecta un obstáculo, se actualiza la KB
        o_lit = Literal(f"O_{fx}_{fy}")
        if sensor_frente:
            self.kb.update_fact(o_lit)
        else:
            self.kb.update_fact(o_lit.negate())

        # Si el sensor de piso detecta suciedad, se actualiza la KB
        l_lit = Literal(f"L_{self.x}_{self.y}")
        if sucio:
            self.kb.update_fact(l_lit.negate())
            self.known_dirt.add((self.x, self.y))
        else:
            self.kb.update_fact(l_lit)
            if (self.x, self.y) in self.known_dirt:
                self.known_dirt.remove((self.x, self.y))

        # Axioma de supervivencia: estoy vivo aquí → no hay alfombra
        self.kb.update_fact(Literal(f"C_{self.x}_{self.y}").negate())

        # Sensor de pelusa (Brisa del Wumpus World)
        p_lit = Literal(f"P_{self.x}_{self.y}")
        if pelusa:
            self.kb.update_fact(p_lit)
            self.pelusa_positions.add((self.x, self.y))
        else:
            self.kb.update_fact(p_lit.negate())

        # Regla bicondicional: P_x_y ⟺ C_adj₁ ∨ C_adj₂ ∨ C_adj₃ ∨ C_adj₄
        # Solo se inyecta una vez por celda visitada
        if (self.x, self.y) not in self._pelusa_rules_added:
            self._pelusa_rules_added.add((self.x, self.y))
            adj = [(self.x+1, self.y), (self.x-1, self.y),
                   (self.x, self.y+1), (self.x, self.y-1)]
            # Excluir vecinos que ya se sabe que son muros
            valid_adj = [(nx, ny) for nx, ny in adj
                         if Literal(f"O_{nx}_{ny}").to_sympy() not in self.kb.clauses]
            if valid_adj:
                sym_P = symbols(f"P_{self.x}_{self.y}")
                sym_C = [symbols(f"C_{ax}_{ay}") for ax, ay in valid_adj]
                self.kb.add_rule(Equivalent(sym_P, Or(*sym_C)))

                # Axioma de separación (perímetro completo, 8 vecinos):
                # Dos alfombras nunca están adyacentes ni en diagonal
                all_8_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]
                if not hasattr(self, '_separation_added'):
                    self._separation_added = set()
                for ax, ay in valid_adj:
                    for dx, dy in all_8_dirs:
                        nax, nay = ax + dx, ay + dy
                        if (nax, nay) != (self.x, self.y):
                            pair = tuple(sorted([(ax, ay), (nax, nay)]))
                            if pair not in self._separation_added:
                                self._separation_added.add(pair)
                                sym_a = symbols(f"C_{ax}_{ay}")
                                sym_b = symbols(f"C_{nax}_{nay}")
                                self.kb.add_rule(Not(And(sym_a, sym_b)))

                # Inferencia inmediata: si los muros reducen los candidatos,
                # intentar deducir la alfombra ahora mismo
                for ax, ay in valid_adj:
                    if (ax, ay) not in self.known_carpets:
                        is_carpet, _ = self.kb.entails(Literal(f"C_{ax}_{ay}"))
                        if is_carpet:
                            self.known_carpets.add((ax, ay))

    # Wrappers de módulos externos

    def _bfs_path(self, start, goal):
        """Usa el BFS del módulo navigation."""
        return bfs_path(start, goal, self._get_valid_adjacents, self.visited, False)

    def _path_to_actions(self, path_coords, look_only_at_end=False):
        """Convierte coordenadas a acciones usando el módulo navigation."""
        return path_to_actions(path_coords, self.x, self.y, self.orientacion, look_only_at_end)

    def _genetic_algorithm_tsp(self, start, targets):
        """Usa el algoritmo genético del módulo genetic."""
        def dist_func(n1, n2):
            if (n1, n2) not in self.dist_cache:
                d = len(self._bfs_path(n1, n2))
                self.dist_cache[(n1, n2)] = d
                self.dist_cache[(n2, n1)] = d
            return self.dist_cache.get((n1, n2), 0)
        return genetic_algorithm_tsp(start, targets, dist_func)

    # Motor de razonamiento

    def think_and_act(self):
        """
        Función principal de razonamiento que decide qué hacer a continuación.
        """
        # Actualizar mapa de calor
        self.visit_count[(self.x, self.y)] += 1
        
        explicacion_total = ""

        while True:
            # Cuestionamiento. ¿Se ha terminado de limpiar y estamos en la base?
            if self.x == self.base[0] and self.y == self.base[1] and self.mode == "GO_FINISH":
                return "TERMINAR", None, explicacion_total + "He llegado a la base tras finalizar la tarea."

            # Cuestionamiento. ¿Estamos en la base y debemos recargar?
            if self.x == self.base[0] and self.y == self.base[1] and self.mode == "GO_CHARGE":
                self.battery = self.max_battery
                self.mode = getattr(self, 'prev_mode', 'EXPLORE')
                return "RECARGAR", None, explicacion_total + "Batería recargada al 100%. Vuelvo a mi tarea anterior."

            # Cuestionamiento ML. ¿Necesito recargar?
            path_coords = self._bfs_path((self.x, self.y), self.base)
            actions_to_base = self._path_to_actions(path_coords)
            cost_to_base = len(actions_to_base)

            battery_pct = self.battery / self.max_battery
            exp_done = 1 if self.mode == "GO_FINISH" or self.mode == "PATROL" else 0 # Se terminó de limpiar?
            pat_done = 1 if not getattr(self, 'patrol_targets', set()) and not getattr(self, 'patrol_route', []) else 0 # Se terminó de patrullar?

            predicted_mode = predict_mode(self.ml_model, battery_pct, cost_to_base, exp_done, pat_done)

            # Toma de decisiones
            # Si el ML nos dice que debemos recargar y no estamos en la base y no estamos ya recargando
            if predicted_mode == "GO_CHARGE" and self.mode in ["EXPLORE", "PATROL"] and (self.x, self.y) != self.base:
                # Si estábamos patrullando, guardamos el lugar para no olvidarlo
                if self.mode == "PATROL" and hasattr(self, 'current_target') and self.current_target:
                    self.patrol_route.insert(0, self.current_target)
                    self.current_target = None
                    
                self.prev_mode = self.mode
                self.mode = "GO_CHARGE"
                self.path = actions_to_base
                explicacion_total += f"Batería crítica. Decision ML: huir a la base.\n"

            # Cuestionamiento. ¿Hay basura donde estoy?
            if self.mode in ["EXPLORE", "PATROL"]:
                query_sucio = Literal(f"L_{self.x}_{self.y}").negate()
                is_dirty, expl = self.kb.entails(query_sucio)
                if is_dirty:
                    self.kb.update_fact(Literal(f"L_{self.x}_{self.y}"))
                    if hasattr(self, 'known_dirt') and (self.x, self.y) in self.known_dirt:
                        self.known_dirt.remove((self.x, self.y))
                    self.battery -= 1
                    explicacion_total += expl + f"\nBasura detectada en ({self.x}, {self.y}) Aspirando...\n"
                    return "ASPIRAR", None, explicacion_total

            # Actuadores. Movimiento y rotación del plan generado
            if self.path:
                action = self.path.pop(0)
                if action == "ROTAR_IZQ":
                    dirs = ["N", "E", "S", "W"]
                    self.orientacion = dirs[(dirs.index(self.orientacion) - 1) % 4]
                    self.battery -= 1
                    return "ROTAR_IZQ", None, explicacion_total + f"[RUTA] Rotando a la izquierda. Mirando a {self.orientacion}"
                elif action == "ROTAR_DER":
                    dirs = ["N", "E", "S", "W"]
                    self.orientacion = dirs[(dirs.index(self.orientacion) + 1) % 4]
                    self.battery -= 1
                    return "ROTAR_DER", None, explicacion_total + f"[RUTA] Rotando a la derecha. Mirando a {self.orientacion}"
                elif isinstance(action, tuple) and action[0] == "MOVER":
                    next_pos = action[1]
                    
                    # Caso especial. Generación espontanea de obstáculos.
                    is_obs, _ = self.kb.entails(Literal(f"O_{next_pos[0]}_{next_pos[1]}"))
                    if is_obs:
                        self.path = [] 
                        explicacion_total += f"\nEl usuario puso un muro en {next_pos}! Calculando ruta nueva."
                        continue
                    
                    self.prev_x, self.prev_y = self.x, self.y
                    self.x, self.y = next_pos
                    self.visited.add((next_pos[0], next_pos[1]))
                    if (self.x, self.y) in self.patrol_targets:
                        self.patrol_targets.remove((self.x, self.y))
                    if hasattr(self, 'patrol_route') and (self.x, self.y) in self.patrol_route:
                        self.patrol_route.remove((self.x, self.y))
                    self.battery -= 1
                    explicacion_total += f"[RUTA] Avanzando a {next_pos}. Batería restante: {self.battery}."
                    return "MOVER", next_pos, explicacion_total

            # Planificación del patrullaje. AG
            if self.mode == "PATROL":
                if self.patrol_targets:
                    explicacion_total += "Creando la ruta para vigilar...\n"
                    self.patrol_route = self._genetic_algorithm_tsp((self.x, self.y), list(self.patrol_targets))
                    self.patrol_targets = set()

                if hasattr(self, 'patrol_route') and self.patrol_route:
                    target = self.patrol_route.pop(0)
                    self.current_target = target
                    path_coords = self._bfs_path((self.x, self.y), target)
                    self.path = self._path_to_actions(path_coords)
                    continue
                else:
                    self.mode = "GO_FINISH"
                    path_coords = self._bfs_path((self.x, self.y), self.base)
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += "Termino el patrullaje. Regreso a la base."

                    if not self.path:
                        if (self.x, self.y) == self.base:
                            return "TERMINAR", None, explicacion_total + " Ya estoy en la base."
                    else:
                        continue
            
            # Recordar dónde se dejó basura.
            if self.mode == "EXPLORE" and getattr(self, 'known_dirt', set()):
                dirt_targets = []
                for d in self.known_dirt:
                    path_coords = self._bfs_path((self.x, self.y), d)
                    if path_coords:
                        dirt_targets.append((d, len(path_coords), path_coords))

                if dirt_targets:
                    dirt_targets.sort(key=lambda x: x[1])
                    target, dist, path_coords = dirt_targets[0]
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += f"\n[MEMORIA] Recordé que dejé basura en {target}. Voy para allá.\n"
                    continue

            # Primera iteración. Reconocimiento del entorno.
            if self.mode == "EXPLORE":
                frontier = set()
                for vx, vy in self.visited:
                    for nx, ny in self._get_valid_adjacents(vx, vy):
                        if (nx, ny) not in self.visited:
                            frontier.add((nx, ny))

                # Motor de Inferencia Retroactiva (Alfombras)
                for fx, fy in list(frontier):
                    if (fx, fy) not in self.known_carpets:
                        is_carpet, _ = self.kb.entails(Literal(f"C_{fx}_{fy}"))
                        if is_carpet:
                            self.known_carpets.add((fx, fy))
                            explicacion_total += f"\n[INFERENCIA] Alfombra deducida en ({fx}, {fy})."

                # Filtrar fronteras: quitar alfombras inferidas
                frontier -= self.known_carpets

                safe_frontiers = []
                unknown_frontiers = []
                risky_frontiers = []    # Sospechosas de alfombra

                for fx, fy in frontier:
                    query_safe = Literal(f"O_{fx}_{fy}").negate()
                    is_safe, expl = self.kb.entails(query_safe)

                    query_obstacle = Literal(f"O_{fx}_{fy}")
                    is_obs, _ = self.kb.entails(query_obstacle)

                    if is_obs:
                        continue

                    # ¿Es adyacente a alguna celda con pelusa?
                    is_near_pelusa = any(
                        (fx + dx, fy + dy) in self.pelusa_positions
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )

                    if is_safe:
                        if not is_near_pelusa:
                            # Sin obstáculo y sin pelusa cerca → seguro
                            dist = len(self._bfs_path((self.x, self.y), (fx, fy)))
                            safe_frontiers.append(((fx, fy), dist, expl))
                        else:
                            # Cerca de pelusa → verificar con KB si probado ~C
                            is_carpet_clear, _ = self.kb.entails(Literal(f"C_{fx}_{fy}").negate())
                            if is_carpet_clear:
                                dist = len(self._bfs_path((self.x, self.y), (fx, fy)))
                                safe_frontiers.append(((fx, fy), dist, expl))
                            else:
                                dist = len(self._bfs_path((self.x, self.y), (fx, fy)))
                                risky_frontiers.append(((fx, fy), dist))
                    elif not is_obs:
                        dist = len(self._bfs_path((self.x, self.y), (fx, fy)))
                        unknown_frontiers.append(((fx, fy), dist))

                if safe_frontiers:
                    safe_frontiers.sort(key=lambda x: x[1])
                    target, dist, expl = safe_frontiers[0]
                    path_coords = self._bfs_path((self.x, self.y), target)
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += f"\nSe demostró matemáticamente que {target} es seguro.\n{expl}"
                    continue
                elif unknown_frontiers:
                    unknown_frontiers.sort(key=lambda x: x[1])
                    target, dist = unknown_frontiers[0]
                    path_coords = self._bfs_path((self.x, self.y), target)
                    self.path = self._path_to_actions(path_coords, look_only_at_end=True)
                    explicacion_total += f"\nNo sé qué hay en {target}. Me acerco a investigar.\n"

                    if not self.path:
                        return "ESPERAR", None, explicacion_total + "Me acerco a investigar..."
                    continue
                elif risky_frontiers:
                    # Último recurso: explorar celda sospechosa de alfombra
                    risky_frontiers.sort(key=lambda x: x[1])
                    target, dist = risky_frontiers[0]
                    path_coords = self._bfs_path((self.x, self.y), target)
                    self.path = self._path_to_actions(path_coords, look_only_at_end=True)
                    explicacion_total += f"\n[RIESGO] No queda otra opción, me acerco a {target} con precaución.\n"

                    if not self.path:
                        return "ESPERAR", None, explicacion_total + "Evaluando riesgo..."
                    continue
                else:
                    self.mode = "GO_FINISH"
                    path_coords = self._bfs_path((self.x, self.y), self.base)
                    self.path = self._path_to_actions(path_coords)
                    explicacion_total += "Reconocimiento completado. Regreso a la base.\n"

                    if not self.path:
                        if (self.x, self.y) == self.base:
                            return "TERMINAR", None, explicacion_total + " Ya estoy en la base."
                    else:
                        continue

            return ("ESPERAR", None, "Pensando...")

    def get_planned_route(self):
        """
        Le pasa su plan a PyGame para que dibuje la línea azul claro.
        """
        route = []
        for action in self.path:
            if isinstance(action, tuple) and action[0] == "MOVER":
                route.append(action[1])
        return route
