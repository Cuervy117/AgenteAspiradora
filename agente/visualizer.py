import pygame

# Caja de colores (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)       # Color de los muros de ladrillo
GREEN = (34, 139, 34)       # Color de la base de recarga 
BLUE = (0, 0, 255)          # Color de la aspiradora (batería llena)
YELLOW = (255, 215, 0)      # Color de la suciedad.
RED = (255, 0, 0)           # Color de la aspiradora cuando la bateria es baja.
DARK_GRAY = (40, 40, 40)    # Color de lo desconocido (Niebla de guerra)

# Medidas
CELL_SIZE = 80              
MARGIN = 5                  
TOP_MARGIN = 40


class Visualizer:
    """
    La clase encargada de dibujar la simulación.
    """

    def __init__(self, width, height):
        """
        Inicialización de la ventana
        """
        pygame.init() # Encendemos el motor gráfico
        self.width = width
        self.height = height

        self.grid_width = width * CELL_SIZE + (width + 1) * MARGIN
        self.grid_height = height * CELL_SIZE + (height + 1) * MARGIN

        self.screen_width = self.grid_width * 2
        self.screen_height = self.grid_height + 350 + TOP_MARGIN

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Aspiradora Autónoma: Entorno Real vs Visión del Agente")
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)

    #  Renderizado principal

    def draw(self, env, agent, paso, action_text, agent_real_x, agent_real_y, percepts=None, explicacion="", action_history=None, show_heatmap=False):
        """
        Dibuja un frame completo de la simulación.
        
        Renderiza: títulos, ambas cuadrículas, separador, panel de info
        y barra de batería con gradiente rojo↔verde.
        """
        self.screen.fill(BLACK)

        # Títulos de cada panel
        title1 = self.title_font.render("Entorno Real", True, WHITE)
        title2 = self.title_font.render("Visión del Agente (Niebla)", True, WHITE)
        self.screen.blit(title1, (self.grid_width // 2 - title1.get_width() // 2, 5))
        self.screen.blit(title2, (self.grid_width + self.grid_width // 2 - title2.get_width() // 2, 5))

        # Dibujar ambas cuadrículas
        self._draw_grid(env, agent, 0, TOP_MARGIN, False, agent_real_x, agent_real_y, show_heatmap)
        self._draw_grid(env, agent, self.grid_width, TOP_MARGIN, True, agent_real_x, agent_real_y, show_heatmap)

        # Separador central
        pygame.draw.line(self.screen, WHITE, (self.grid_width, 0), (self.grid_width, self.screen_height - 250), 2)
        pygame.draw.line(self.screen, WHITE, (0, self.grid_height + TOP_MARGIN + 10), (self.screen_width, self.grid_height + TOP_MARGIN + 10), 2)

        # Panel de información inferior
        info_y = self.grid_height + TOP_MARGIN + 20
        
        text_pos = self.font.render(f"Paso: {paso} | Relativa: ({agent.x}, {agent.y}) | Real: ({agent_real_x}, {agent_real_y}) | Dir: {agent.orientacion} | Modo: {agent.mode}", True, WHITE)
        self.screen.blit(text_pos, (10, info_y))
        
        if percepts:
            p_fr = "SÍ" if percepts.get("Obstaculo_Frente") else "NO"
            p_su = "SÍ" if percepts.get("Suciedad") else "NO"
            p_ch = "SÍ" if percepts.get("Colision") else "NO"
            p_pe = "SÍ" if percepts.get("Pelusa") else "NO"
            perc_str = f"Frente:{p_fr} | Sucio:{p_su} | Choque:{p_ch} | Pelusa:{p_pe}"
        else:
            perc_str = "Ninguna"
            
        text_perc = self.font.render(f"Percepciones: {perc_str}", True, YELLOW)
        self.screen.blit(text_perc, (10, info_y + 30))
        
        text_action = self.font.render(f"Acción: {action_text}", True, (100, 255, 100))
        self.screen.blit(text_action, (10, info_y + 60))
        
        expl_y = info_y + 90
        text_expl_title = self.font.render("Cadena de Inferencia:", True, WHITE)
        self.screen.blit(text_expl_title, (10, expl_y))
        
        small_font = pygame.font.SysFont("Arial", 16)
        if explicacion:
            clip_rect = pygame.Rect(10, expl_y + 25, self.grid_width - 20, 200)
            self.screen.set_clip(clip_rect)
            lines = explicacion.split('\n')
            draw_idx = 0
            for line in lines:
                if line.strip():
                    text_line = small_font.render(line.strip(), True, (200, 200, 200))
                    self.screen.blit(text_line, (20, expl_y + 25 + draw_idx * 20))
                    draw_idx += 1
            self.screen.set_clip(None) 
                    
        # Panel de Historial de Acciones
        hist_title = self.font.render("Historial de Acciones:", True, WHITE)
        self.screen.blit(hist_title, (self.grid_width + 10, expl_y))
        if action_history:
            hist_y = expl_y + 25
            for i, act_str in enumerate(reversed(action_history)):
                color_val = max(100, 255 - i * 40)
                h_text = small_font.render(f"- {act_str}", True, (color_val, color_val, color_val))
                self.screen.blit(h_text, (self.grid_width + 20, hist_y + i * 20))

        # Barra de batería con gradiente de color
        battery_pct = max(0, min(1.0, agent.battery / agent.max_battery))
        r = min(255, int(255 * 2 * (1 - battery_pct)))
        g = min(255, int(255 * 2 * battery_pct))
        battery_color = (r, g, 0)

        bar_width = 150
        bar_height = 25
        bar_x = self.screen_width - bar_width - 15
        bar_y = info_y

        bat_label = self.font.render("Batería:", True, WHITE)
        self.screen.blit(bat_label, (bar_x - bat_label.get_width() - 10, bar_y + 2))

        fill_width = int(bar_width * battery_pct)
        if fill_width > 0:
            pygame.draw.rect(self.screen, battery_color, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)

        bat_text = self.font.render(f"{int(battery_pct*100)}%", True, BLACK if 0.3 < battery_pct < 0.7 else WHITE)
        text_rect = bat_text.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.screen.blit(bat_text, text_rect)
        
        # Estadísticas de Eficiencia 
        stats_y = bar_y + 40
        explored = len(agent.visited)
        total_cells = (env.width * env.height) - len(env.obstacles) - len(getattr(env, 'carpets', set()))
        efficiency = round(env.active_cleaning_steps / max(1, env.dirt_cleaned), 1) if env.dirt_cleaned > 0 else 0
        restante = len(env.dirt)
        
        stat_text = self.font.render(f"Limpiado: {env.dirt_cleaned} | Pendiente: {restante} | Eficiencia: {efficiency} pasos/suc | Exploración: {explored}/{total_cells}", True, (200, 200, 255))
        self.screen.blit(stat_text, (self.screen_width - stat_text.get_width() - 15, stats_y))

        pygame.display.flip()

    #  Renderizado de la cuadrícula 

    def _draw_grid(self, env, agent, offset_x, offset_y, is_agent_view, agent_real_x, agent_real_y, show_heatmap):
        """
        Dibuja una cuadrícula completa (panel real o panel del agente).
        
        Para la vista del agente, se realiza un mapeo de coordenadas:
          ax = x_pantalla - start_real_x   (convierte coord. real → relativa)
        Esto permite alinear la perspectiva del agente con el panel real.
        
        Lógica de renderizado de la vista del agente (fog of war):
          - Casilla visitada → Blanca (explorada)
          - Base del agente → Verde
          - Obstáculo confirmado (KB |= O(x,y)) → Marrón
          - Casilla desconocida → Gris oscuro (niebla)
          - Suciedad conocida (KB |= ¬L(x,y)) → Punto amarillo
          - Casilla en patrullaje pendiente → Azul-grisáceo
        """
        for y in range(self.height):
            for x in range(self.width):
                draw_y = self.height - 1 - y
                rect = pygame.Rect(
                    offset_x + MARGIN + x * (CELL_SIZE + MARGIN),
                    offset_y + MARGIN + draw_y * (CELL_SIZE + MARGIN),
                    CELL_SIZE, CELL_SIZE
                )

                color = WHITE
                has_dirt = False
                is_fog = False

                if not is_agent_view:
                    if env.is_obstacle(x, y):
                        color = BROWN
                    elif hasattr(env, 'base') and (x, y) == env.base:
                        color = GREEN
                    elif hasattr(env, 'carpets') and (x, y) in env.carpets:
                        color = (180, 0, 0) # Rojo oscuro para la alfombra
                        
                    if (x, y) in env.dirt:
                        has_dirt = True
                else:
                    # Panel AGENTE: mapeo de coordenadas y fog of war
                    # Convertir coordenada de pantalla a coordenada relativa del agente
                    ax = x - getattr(env, 'start_real_x', 0)
                    ay = y - getattr(env, 'start_real_y', 0)

                    is_wall = False
                    if hasattr(agent, 'kb'):
                        from logic import Literal
                        obs_sym = Literal(f"O_{ax}_{ay}").to_sympy()
                        if obs_sym in agent.kb.clauses:
                            is_wall = True
                    else:
                        is_wall = (ax, ay) in getattr(agent, 'walls', set())

                    if is_wall:
                        color = BROWN
                        is_fog = False
                    else:
                        risk = agent.carpet_prob.get((ax, ay), 0.0) if hasattr(agent, 'carpet_prob') else 0
                        is_carpet = (ax, ay) in getattr(agent, 'known_carpets', set()) or risk >= 0.9

                        if is_carpet:
                            color = (180, 0, 0)
                            is_fog = False
                        elif risk >= 0.5:
                            color = (255, 140, 0)  # Naranja
                            is_fog = False
                        elif risk > 0.0:
                            color = (200, 180, 50) # Amarillo/marrón claro
                            is_fog = False
                        elif (ax, ay) in agent.visited:
                            color = WHITE
                            if is_agent_view and show_heatmap:
                                visits = agent.visit_count.get((ax, ay), 1)
                                intensity = min(255, visits * 25)
                                color = (255 - intensity, 255 - int(intensity * 0.5), 255)
                            if hasattr(agent, 'base') and (ax, ay) == agent.base:
                                color = GREEN

                            in_targets = hasattr(agent, 'patrol_targets') and (ax, ay) in agent.patrol_targets
                            in_route = hasattr(agent, 'patrol_route') and (ax, ay) in agent.patrol_route
                            if in_targets or in_route:
                                color = (70, 90, 110)
                                is_fog = True
                        else:
                            color = DARK_GRAY
                            is_fog = True
                    # Suciedad conocida
                    if (ax, ay) in getattr(agent, 'known_dirt', set()) or (ax, ay) in getattr(agent, 'dirt_cells', set()):
                        has_dirt = True

                pygame.draw.rect(self.screen, color, rect)

                if has_dirt and not is_fog:
                    pygame.draw.circle(self.screen, YELLOW, rect.center, CELL_SIZE // 4)

                # Dibujar Pelusa 'P'
                has_pelusa = False
                if not env.is_obstacle(x, y):
                    if not is_agent_view:
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            if hasattr(env, 'carpets') and (x + dx, y + dy) in env.carpets:
                                has_pelusa = True
                                break
                    else:
                        # Vista del agente: mostrar pelusa si la posición la detectó
                        if (ax, ay) in getattr(agent, 'pelusa_positions', set()):
                            has_pelusa = True
                
                if has_pelusa and not is_fog:
                    p_text = self.font.render("P", True, (255, 100, 200))
                    p_rect = p_text.get_rect(center=(rect.centerx + CELL_SIZE//3, rect.centery - CELL_SIZE//3))
                    self.screen.blit(p_text, p_rect)

                # Dibujar el agente (círculo + flecha de orientación)
                is_agent_pos = False
                if not is_agent_view and (x, y) == (agent_real_x, agent_real_y):
                    is_agent_pos = True
                elif is_agent_view:
                    ax = x - getattr(env, 'start_real_x', 0)
                    ay = y - getattr(env, 'start_real_y', 0)
                    if (ax, ay) == (agent.x, agent.y):
                        is_agent_pos = True

                if is_agent_pos:
                    agent_color = BLUE if agent.battery > 5 else RED
                    pygame.draw.circle(self.screen, agent_color, rect.center, CELL_SIZE // 2 - 10)

                    # Flecha indicadora de orientación
                    center_x, center_y = rect.center
                    radius = CELL_SIZE // 2 - 10
                    if getattr(agent, 'orientacion', 'N') == "N":
                        end_pos = (center_x, center_y - radius)
                    elif agent.orientacion == "S":
                        end_pos = (center_x, center_y + radius)
                    elif agent.orientacion == "E":
                        end_pos = (center_x + radius, center_y)
                    else:
                        end_pos = (center_x - radius, center_y)

                    pygame.draw.line(self.screen, BLACK, rect.center, end_pos, 4)

        # Dibujar la ruta planificada
        if is_agent_view:
            route = agent.get_planned_route()
            if route:
                start_real_x = getattr(env, 'start_real_x', 0)
                start_real_y = getattr(env, 'start_real_y', 0)
                
                points = []
                curr_screen_x = agent.x + start_real_x
                curr_screen_y = agent.y + start_real_y
                draw_curr_y = self.height - 1 - curr_screen_y
                points.append((offset_x + MARGIN + curr_screen_x * (CELL_SIZE + MARGIN) + CELL_SIZE // 2,
                               offset_y + MARGIN + draw_curr_y * (CELL_SIZE + MARGIN) + CELL_SIZE // 2))
                
                for rx, ry in route:
                    screen_x = rx + start_real_x
                    screen_y = ry + start_real_y
                    draw_y = self.height - 1 - screen_y
                    px = offset_x + MARGIN + screen_x * (CELL_SIZE + MARGIN) + CELL_SIZE // 2
                    py = offset_y + MARGIN + draw_y * (CELL_SIZE + MARGIN) + CELL_SIZE // 2
                    points.append((px, py))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, (0, 255, 255), False, points, 4)


    def process_events(self):
        """
        Procesa eventos de PyGame (cierre de ventana y clics del usuario).
        
        Permite interactividad en tiempo real:
          - Clic izquierdo en el panel real: añade suciedad.
          - Clic derecho en el panel real: añade obstáculo.
        
        Returns:
            tuple: (running, clicks_list) donde clicks_list contiene
                   tuplas (button, grid_x, grid_y).
        """
        clicks = []
        key_events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, clicks, key_events
            elif event.type == pygame.KEYDOWN:
                key_events.append(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if MARGIN <= mx <= self.grid_width - MARGIN and TOP_MARGIN <= my <= self.grid_height + TOP_MARGIN:
                    grid_x = (mx - MARGIN) // (CELL_SIZE + MARGIN)
                    draw_y = (my - TOP_MARGIN - MARGIN) // (CELL_SIZE + MARGIN)
                    grid_y = self.height - 1 - draw_y
                    clicks.append((event.button, grid_x, grid_y))
        return True, clicks, key_events

    def close(self):
        """Cierra la ventana de PyGame y libera recursos."""
        pygame.quit()
