"""
Lógica dedicada al comportamiento del entorno.

Se definen las interacciones del agente con el mundo. En este archivo se definen las colisiones, 
la suciedad y la base.

No se separó la lógica del actuador y sensores por temas de prácticidad.
"""

import random

class Environment:
    """
    La clase que construye y mantiene las reglas del entorno.
    """

    def __init__(self, width=6, height=6, num_obstacles=8, num_dirt=10, start_base=None, num_carpets=2):
        """
        Inicializa el entorno.
        
        Args:
            width: El ancho del tablero.
            height: El alto del tablero.
            num_obstacles: Cuántos muros generar.
            num_dirt: Cuánta suciedad generar al inicio.
            start_base: Dónde poner la estación de carga.
            num_carpets: Cuántas alfombras peligrosas generar.
        """
        self.width = width
        self.height = height
        self.obstacles = set() 
        self.dirt = set()       
        self.total_dirt_spawned = 0 
        self.dirt_cleaned = 0       
        self.active_cleaning_steps = 0
        self.carpets = set()

        if start_base is None:
            self.base = (random.randint(0, width-1), random.randint(0, height-1))
        else:
            self.base = start_base

        # Coordenadas reales del agente en el entorno global
        self.agent_real_x = self.base[0]
        self.agent_real_y = self.base[1]
        self.start_real_x = self.base[0]
        self.start_real_y = self.base[1]

        base_y_adyacentes = {
            self.base,
            (self.base[0]+1, self.base[1]),
            (self.base[0]-1, self.base[1]),
            (self.base[0], self.base[1]+1),
            (self.base[0], self.base[1]-1)
        }

        # Generación de obstáculos
        while len(self.obstacles) < num_obstacles:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if (x, y) not in base_y_adyacentes:
                self.obstacles.add((x, y))

        # Generación de alfombras (nunca adyacentes entre sí)
        carpet_attempts = 0
        while len(self.carpets) < num_carpets and carpet_attempts < 200:
            carpet_attempts += 1
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if (x, y) not in base_y_adyacentes and (x, y) not in self.obstacles:
                # Verificar que no sea adyacente (ni diagonal) a otra alfombra
                adjacent_to_carpet = any(
                    (x + dx, y + dy) in self.carpets
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
                )
                if not adjacent_to_carpet:
                    self.carpets.add((x, y))

        self.spawn_dirt(num_dirt)

    # Lógica para interactuar con el entorno

    def remove_dirt(self, x, y):
        """
        Remueve la basura de la posición dada.

        Args:
            x: Coordenada x
            y: Coordenada y
        
        Returns:
            None
        """
        if (x, y) in self.dirt:
            self.dirt.remove((x, y))
            self.dirt_cleaned += 1

    def spawn_dirt(self, num_dirt=5):
        """
        Genera nuevas partículas de suciedad en posiciones aleatorias y alcanzables.

        Args:
            num_dirt: Cantidad de partículas de suciedad a generar
        
        Returns:
            None
        """
        spawned = 0
        attempts = 0
        max_attempts = 100
        while spawned < num_dirt and attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) != self.base and (x, y) not in self.obstacles and (x, y) not in self.dirt and (x, y) not in getattr(self, 'carpets', set()):
                if self._is_reachable((x, y)):
                    self.dirt.add((x, y))
                    self.total_dirt_spawned += 1
                    spawned += 1
            attempts += 1

    # Función auxiliar para verificar si una posición es alcanzable
    def _is_reachable(self, goal):
        """
        Determina si una posición es alcanzable desde la base.

        Args:
            goal: Coordenada (x, y) a verificar
        
        Returns:
            True si es alcanzable, False en caso contrario
        """
        queue = [self.base]
        visited = {self.base}
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) == goal:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in getattr(self, 'carpets', set()) and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False
        
    # Lógica de sensores

    def is_obstacle(self, x, y):
        """
        Determina si en la posición dada hay un obstáculo.
        
        Args:
            x: Coordenada x
            y: Coordenada y
        
        Returns:
            True si hay un obstáculo, False en caso contrario
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return (x, y) in self.obstacles

    def get_percepts(self, agent_x, agent_y, orientacion, agent_real_x, agent_real_y):
        """
        Recibe la posicion y orientacion del agente, y le devuelve
        un diccionario con la informacion que necesita para tomar decisiones.

        Args:
            agent_x: Posición x del agente
            agent_y: Posición y del agente
            orientacion: Orientación del agente ("N", "S", "E", "W")
            agent_real_x: Posición real x del agente
            agent_real_y: Posición real y del agente
        
        Returns:
            "Obstaculo_Frente": True o False.
            "Suciedad": True o False.
            "Colision": True o False.
            "Pelusa": True o False.
        """
        is_dirty = (agent_real_x, agent_real_y) in self.dirt

        # Determinar si hay alfombras adyacentes (Pelusa)
        pelusa = False
        for ady_x, ady_y in [(agent_real_x+1, agent_real_y), (agent_real_x-1, agent_real_y), 
                             (agent_real_x, agent_real_y+1), (agent_real_x, agent_real_y-1)]:
            if (ady_x, ady_y) in self.carpets:
                pelusa = True
                break

        # Determinamos la casilla frontal
        fx, fy = agent_real_x, agent_real_y
        if orientacion == "N": fy += 1
        elif orientacion == "S": fy -= 1
        elif orientacion == "E": fx += 1
        elif orientacion == "W": fx -= 1

        sensor_frente = self.is_obstacle(fx, fy)

        colision = self.is_obstacle(agent_real_x, agent_real_y)
        
        return {"Obstaculo_Frente": sensor_frente, "Suciedad": is_dirty, "Colision": colision, "Pelusa": pelusa}