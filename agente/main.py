"""
=============================================================================
  main.py — Bucle principal de la simulación
=============================================================================
  Orquesta el ciclo Percibir → Razonar → Actuar del agente aspiradora,
  coordinando el entorno, el agente y la visualización.
  
  Implementa el "Programa del Entorno" (Russell & Norvig §2.4, Fig. 2.13)
  que ejecuta la simulación paso a paso, y el sistema de coordenadas
  duales donde el agente cree iniciar en (0,0) mientras que en la
  realidad está en una posición aleatoria del mapa.
=============================================================================
"""

import random
import collections
from environment import Environment
from agent import VacuumAgent
from visualizer import Visualizer, MARGIN, TOP_MARGIN, CELL_SIZE
import pygame


def main():
    """
    Función principal del programa.

    1. Se crea el entorno, con sus obstaculos y suciedad.
    2. Se crea el agente, con su nivel de bateria y posicion inicial.
    3. Se crea el visualizador, que nos mostrara el entorno y el agente.
    4. Se inicia el bucle principal, que se ejecutara hasta que el usuario cierre el programa o se agote la bateria del agente.

    Nota: viz.draw dibuja todo el entorno, el agente, el paso actual, el texto de la accion, la explicacion de la accion, el historial de acciones y el mapa de calor.
    """
    
    env_width = 8
    env_height = 8

    # Creamos un mundo aleatorio: entre 10 y 20 paredes (obstáculos) y entre 5 y 10 basuras.
    env = Environment(width=env_width, height=env_height, num_obstacles=random.randint(10, 20), num_dirt=random.randint(5, 10), num_carpets=2)

    # El agente siempre inicia en su base (coord. interna 0,0).
    # El visualizador traduce entre el sistema interno (0,0-based) y el mundo real
    # usando env.start_real_x / env.start_real_y como offset.
    agent = VacuumAgent(width=env_width, height=env_height, start_x=0, start_y=0, max_battery=120)
 
    # Iniciamos el entorno gráfico
    viz = Visualizer(width=env_width, height=env_height)

    #  Variables de comportamiento
    paso = 1
    action_text = "Iniciando..."
    running = True          
    recharging = False       
    recharge_counter = 0        
    
    # Variables de control de la simulación
    delay_ms = 166         
    paused = False         
    show_heatmap = True    
    action_history = collections.deque(maxlen=5) 

    # Bucle principal de la simulación
    while running:

        # Listeners de la ventana
        running, clicks, key_events = viz.process_events()
        if not running:
            break  
            
        # Revisamos si presionaron alguna tecla
        for key in key_events:
            if key == pygame.K_UP:
                delay_ms = max(0, delay_ms - 25) # aumentamos la velocidad de la simulación
            elif key == pygame.K_DOWN:
                delay_ms = min(1000, delay_ms + 25) # ralentizamos la simulación
            elif key == pygame.K_SPACE:
                paused = not paused             
            elif key == pygame.K_h:
                show_heatmap = not show_heatmap  
            elif key == pygame.K_c:
                mx, my = pygame.mouse.get_pos()
                if MARGIN <= mx <= viz.grid_width - MARGIN and TOP_MARGIN <= my <= viz.grid_height + TOP_MARGIN:
                    grid_x = (mx - MARGIN) // (CELL_SIZE + MARGIN)
                    draw_y = (my - TOP_MARGIN - MARGIN) // (CELL_SIZE + MARGIN)
                    grid_y = env.height - 1 - draw_y
                    if 0 <= grid_x < env.width and 0 <= grid_y < env.height and getattr(env, 'base', (0, 0)) != (grid_x, grid_y):
                        if (grid_x, grid_y) in env.dirt: env.dirt.remove((grid_x, grid_y))
                        if (grid_x, grid_y) in env.obstacles: env.obstacles.remove((grid_x, grid_y))
                        if not hasattr(env, 'carpets'): env.carpets = set()
                        env.carpets.add((grid_x, grid_y))
                        action_text = f"El usuario puso una alfombra en ({grid_x}, {grid_y})"

        # Revisamos si se realizaron clicks
        for btn, cx, cy in clicks:
            if btn == 1:  # Clic izquierdo
                if 0 <= cx < env.width and 0 <= cy < env.height and (cx, cy) not in env.obstacles and getattr(env, 'base', (0, 0)) != (cx, cy):
                    env.dirt.add((cx, cy)) # Se agrega la suciedad
                    action_text = f"El usuario ensució ({cx}, {cy})"
            elif btn == 3:  # Clic derecho
                if 0 <= cx < env.width and 0 <= cy < env.height and getattr(env, 'base', (0, 0)) != (cx, cy):
                    if (cx, cy) in env.dirt: env.dirt.remove((cx, cy)) # Si había basura bajo el muro, se aplasta
                    env.obstacles.add((cx, cy)) # Se agrega el obstáculo
                    action_text = f"El usuario puso un objeto en ({cx}, {cy})"

        if paused:
            viz.draw(env, agent, paso, "Simulación pausada", env.agent_real_x, env.agent_real_y, None, "", action_history, show_heatmap)
            pygame.time.wait(100) # Esperamos una décima de segundo y volvemos a empezar
            continue

        if recharging:
            recharge_counter += 1
            action_text = f"Recargando... ({recharge_counter}/10)"
            
            if recharge_counter >= 10:
                env.spawn_dirt(random.randint(5, 10))
                agent.mode = "PATROL" 
                agent.patrol_targets = set(agent.visited)
                recharging = False
                action_text = "Nuevo ciclo. Modo de limpieza iniciado."
                action_history.append(action_text)
            
            viz.draw(env, agent, paso, action_text, env.agent_real_x, env.agent_real_y, None, "", action_history, show_heatmap)
            pygame.time.wait(delay_ms)
            if not recharging:
                paso += 1
            continue

    # Agente inteligente

        # Sensores (Percepciones)
        percepts = env.get_percepts(agent.x, agent.y, agent.orientacion, env.agent_real_x, env.agent_real_y)

        agent.perceive(percepts) 

        # Razonamiento
        action, args, expl = agent.think_and_act()

        # Interpretación del razonamiento
        if action == "ASPIRAR":
            action_text = f"Aspirando suciedad en ({agent.x}, {agent.y})"
        elif action == "MOVER":
            action_text = f"Moviendo hacia {args}"
        elif action == "ROTAR_IZQ":
            action_text = "Girando a la izquierda"
        elif action == "ROTAR_DER":
            action_text = "Girando a la derecha"
        elif action == "ESPERAR":
            action_text = "Pensando qué hacer..."
        elif action == "CHOQUE":
            action_text = f"Choque con la pared. Retrocediendo..."
        elif action == "RECARGAR":
            action_text = "Batería al 100%."
        elif action == "TERMINAR":
            action_text = "Limpieza completada. Volviendo a la base."

        # Actuadores
        if action == "ASPIRAR":
            env.remove_dirt(env.agent_real_x, env.agent_real_y) 
        elif action == "MOVER":
            if agent.orientacion == "N": env.agent_real_y += 1
            elif agent.orientacion == "S": env.agent_real_y -= 1
            elif agent.orientacion == "E": env.agent_real_x += 1
            elif agent.orientacion == "W": env.agent_real_x -= 1
            
            if (env.agent_real_x, env.agent_real_y) in getattr(env, 'carpets', set()):
                action_text = "La aspiradora rompió la alfombra."
                action_history.append(action_text)
                viz.draw(env, agent, paso, action_text, env.agent_real_x, env.agent_real_y, percepts, expl, action_history, show_heatmap)
                pygame.time.wait(3000)
                running = False
                continue
        elif action == "TERMINAR":
            recharging = True
            recharge_counter = 0

        action_history.append(action_text)

        viz.draw(env, agent, paso, action_text, env.agent_real_x, env.agent_real_y, percepts, expl, action_history, show_heatmap)

        # Variable para calcular la eficiencia (pasos / basura limpiada)
        if env.dirt:
            env.active_cleaning_steps += 1
        paso += 1
        pygame.time.wait(delay_ms)
    viz.close()

if __name__ == "__main__":
    main()
