# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import heapq
from collections import deque
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions

SOME_DEFAULT_VALUE = 5

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class OffensiveAgent(CaptureAgent):
    """
    Agente ofensivo que busca comida utilizando A* y evita enemigos manteniendo una distancia segura.
    """
    def choose_action(self, game_state):
        """
        Selecciona la acción óptima para recolectar comida mientras evita enemigos.
        """
        # Observa la posición de los enemigos
        enemies = self.get_visible_enemies(game_state)

        # Encuentra la comida más cercana
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        closest_food = min(food_list, key=lambda f: self.get_maze_distance(game_state, my_pos, f))

        # Realiza una búsqueda A* hacia la comida más cercana evitando enemigos
        path = self.a_star_search(game_state, my_pos, closest_food, enemies)
        if path and len(path) > 1:
            next_step = path[1]
            return self.get_direction(my_pos, next_step)

        # Si no se encuentra un camino seguro, elige una acción aleatoria segura
        safe_actions = self.get_safe_actions(game_state, enemies)
        if safe_actions:
            return random.choice(safe_actions)

        return Directions.STOP

    def get_visible_enemies(self, game_state):
        """
        Retorna una lista de posiciones de los enemigos visibles que no están asustados.
        """
        enemies = []
        opponents = self.get_opponents(game_state)
        for opponent in opponents:
            enemy_state = game_state.get_agent_state(opponent)
            if enemy_state is not None and enemy_state.get_position() is not None and not enemy_state.is_pacman and enemy_state.scared_timer == 0:
                enemies.append(enemy_state.get_position())
        return enemies

    def get_safe_actions(self, game_state, enemies):
        """
        Retorna una lista de acciones que mantienen al agente a una distancia segura de los enemigos.
        """
        safe_actions = []
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            pos = successor.get_agent_position(self.index)
            if pos:
                if all(self.get_maze_distance(game_state, pos, enemy) > 2 for enemy in enemies):
                    safe_actions.append(action)
        return safe_actions

    def a_star_search(self, game_state, start, goal, enemies):
        """
        Implementa el algoritmo A* para encontrar el camino más corto desde start hasta goal evitando enemigos.
        """
        open_set = []
        heapq.heappush(open_set, (0 + self.get_maze_distance(game_state, start, goal), 0, start, [start]))
        closed_set = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self.get_successors_positions(game_state, current):
                if neighbor in closed_set:
                    continue
                if any(self.get_maze_distance(game_state, neighbor, enemy) <= 2 for enemy in enemies):
                    continue  # Evita posiciones demasiado cercanas a enemigos
                new_cost = cost + 1
                priority = new_cost + self.get_maze_distance(game_state, neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))

        return []

    def get_successors_positions(self, game_state, position):
        """
        Retorna una lista de posiciones adyacentes accesibles desde la posición dada.
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(position[0] + dx), int(position[1] + dy)
            if not game_state.has_wall(next_x, next_y):
                successors.append((next_x, next_y))
        return successors

    def get_direction(self, current, next_step):
        """
        Retorna la dirección desde current hacia next_step.
        """
        dx = next_step[0] - current[0]
        dy = next_step[1] - current[1]
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP

    def get_maze_distance(self, game_state, pos1, pos2):
        """
        Calcula la distancia en el laberinto entre pos1 y pos2 usando BFS.
        """
        queue = deque()
        queue.append((pos1, 0))
        visited = set()
        visited.add(pos1)

        while queue:
            current_pos, dist = queue.popleft()
            if current_pos == pos2:
                return dist

            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                next_x, next_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
                next_pos = (next_x, next_y)

                if not game_state.has_wall(next_x, next_y) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))

        return float('inf')  # Retorna infinito si no hay camino

class DefensiveAgent(CaptureAgent):
    """
    Agente defensivo que utiliza la búsqueda alfa-beta para anticipar y bloquear movimientos enemigos.
    """
    def choose_action(self, game_state):
        """
        Selecciona la acción óptima para defender utilizando la búsqueda alfa-beta.
        """
        invaders = self.get_invaders(game_state)

        if invaders:
            # Si hay invasores, interceptarlos
            invader = min(invaders, key=lambda inv: self.get_maze_distance(game_state, self.get_current_position(game_state), inv))
            action = self.alpha_beta_search(game_state, depth=2, target=invader)
            return action if action else Directions.STOP
        else:
            # No hay invasores, moverse hacia la posición de defensa seleccionada dinámicamente
            defend_position = self.select_defend_position(game_state)
            action = self.alpha_beta_search(game_state, depth=2, target=defend_position)
            return action if action else Directions.STOP

    def get_invaders(self, game_state):
        """
        Retorna una lista de posiciones de los invasores (enemigos que son Pac-Man).
        """
        invaders = []
        opponents = self.get_opponents(game_state)
        for opponent in opponents:
            state = game_state.get_agent_state(opponent)
            if state.is_pacman and state.get_position() is not None:
                invaders.append(state.get_position())
        return invaders

    def select_defend_position(self, game_state):
        """
        Selecciona dinámicamente una posición en nuestro territorio para defender, basada en la distribución de la comida.
        """
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if food_list:
            # Seleccionar la comida propia más cercana a la frontera
            mid_x = game_state.data.layout.width // 2
            if self.red:
                min_distance = min([abs(food[0] - (mid_x - 1)) for food in food_list])
                defend_food = [food for food in food_list if abs(food[0] - (mid_x - 1)) == min_distance]
            else:
                min_distance = min([abs(food[0] - mid_x) for food in food_list])
                defend_food = [food for food in food_list if abs(food[0] - mid_x) == min_distance]
            # Seleccionar la posición más céntrica verticalmente
            defend_position = min(defend_food, key=lambda f: abs(f[1] - game_state.data.layout.height // 2))
            return defend_position
        else:
            # Si no hay comida, retornar al centro de nuestro territorio
            mid_x = game_state.data.layout.width // 2
            if self.red:
                return (mid_x - 1, game_state.data.layout.height // 2)
            else:
                return (mid_x, game_state.data.layout.height // 2)

    def alpha_beta_search(self, game_state, depth, target=None):
        """
        Implementa la búsqueda alfa-beta para determinar la mejor acción hacia el objetivo.
        """
        def max_value(state, depth, alpha, beta):
            if depth == 0 or state.is_over():
                return self.evaluate(state, target)
            value = float('-inf')
            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                value = max(value, min_value(successor, depth -1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(state, depth, alpha, beta):
            if depth ==0 or state.is_over():
                return self.evaluate(state, target)
            value = float('inf')
            opponents = self.get_opponents(state)
            for opponent in opponents:
                opponent_state = state.get_agent_state(opponent)
                if opponent_state is None or opponent_state.get_position() is None:
                    continue
                for action in state.get_legal_actions(opponent):
                    successor = state.generate_successor(opponent, action)
                    value = min(value, max_value(successor, depth -1, alpha, beta))
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
            return value

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            score = min_value(successor, depth -1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action

    def evaluate(self, game_state, target):
        """
        Evalúa el estado del juego desde la perspectiva defensiva.
        """
        my_pos = game_state.get_agent_position(self.index)
        invaders = self.get_invaders(game_state)
        score = 0

        # Penalización por número de invasores
        score -= 1000 * len(invaders)

        # Minimizar la distancia a los invasores
        if invaders:
            distances = [self.get_maze_distance(game_state, my_pos, invader) for invader in invaders]
            score -= 10 * min(distances)

        # Si no hay invasores, moverse hacia la posición de defensa
        elif target is not None:
            distance_to_target = self.get_maze_distance(game_state, my_pos, target)
            score -= distance_to_target

        return score

    def get_maze_distance(self, game_state, pos1, pos2):
        """
        Calcula la distancia en el laberinto entre pos1 y pos2 usando BFS.
        """
        queue = deque()
        queue.append((pos1, 0))
        visited = set()
        visited.add(pos1)

        while queue:
            current_pos, dist = queue.popleft()
            if current_pos == pos2:
                return dist

            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                next_x, next_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
                next_pos = (next_x, next_y)

                if not game_state.has_wall(next_x, next_y) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))

        return float('inf')  # Retorna infinito si no hay camino

    def get_current_position(self, game_state):
        return game_state.get_agent_position(self.index)
