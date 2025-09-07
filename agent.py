import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import json
from datetime import datetime
import math
from heapq import heappush, heappop

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

class PathSolver:
    @staticmethod
    def shortest_path_a_star(game, target=None):
        """A* algorithm for shortest path to food"""
        if target is None:
            target = game.food
        
        head = game.snake[0]
        grid_width = game.w // BLOCK_SIZE
        grid_height = game.h // BLOCK_SIZE
        
        # Convert points to grid coordinates
        start = (head.x // BLOCK_SIZE, head.y // BLOCK_SIZE)
        goal = (target.x // BLOCK_SIZE, target.y // BLOCK_SIZE)
        
        # Check if goal is valid
        if (goal[0] < 0 or goal[0] >= grid_width or 
            goal[1] < 0 or goal[1] >= grid_height):
            return None
        
        # A* algorithm
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: PathSolver.heuristic(start, goal)}
        
        open_set_hash = {start}
        
        while open_set:
            current = heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Return reversed path
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check boundaries
                if (neighbor[0] < 0 or neighbor[0] >= grid_width or 
                    neighbor[1] < 0 or neighbor[1] >= grid_height):
                    continue
                
                # Check if neighbor is part of snake body (excluding tail)
                neighbor_point = Point(neighbor[0] * BLOCK_SIZE, neighbor[1] * BLOCK_SIZE)
                if neighbor_point in game.snake[:-1]:  # Exclude tail
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + PathSolver.heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None  # No path found

    @staticmethod
    def heuristic(a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def longest_path(game):
        """Find a longer path that avoids dead ends (not truly longest but safer)"""
        path = PathSolver.shortest_path_a_star(game)
        if not path:
            return None
            
        # Add some detours to make path longer/safer
        extended_path = []
        for i in range(len(path)):
            extended_path.append(path[i])
            # Occasionally add small detours if safe
            if i < len(path) - 1 and random.random() < 0.2:
                current = path[i]
                next_cell = path[i + 1]
                
                # Find a safe detour direction
                dx, dy = next_cell[0] - current[0], next_cell[1] - current[1]
                detour_dirs = [(dy, -dx), (-dy, dx)]  # Perpendicular directions
                
                for detour_dir in detour_dirs:
                    detour_cell = (current[0] + detour_dir[0], current[1] + detour_dir[1])
                    
                    # Check if detour is valid
                    if (0 <= detour_cell[0] < game.w // BLOCK_SIZE and 
                        0 <= detour_cell[1] < game.h // BLOCK_SIZE):
                        detour_point = Point(detour_cell[0] * BLOCK_SIZE, detour_cell[1] * BLOCK_SIZE)
                        if detour_point not in game.snake:
                            extended_path.append(detour_cell)
                            extended_path.append(current)  # Return to path
                            break
        
        return extended_path

    @staticmethod
    def greedy_direction(game):
        """Greedy approach - move toward food while avoiding obstacles"""
        head = game.snake[0]
        food = game.food
        
        # Get possible safe directions
        possible_directions = []
        for direction in [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]:
            # Check if this direction is safe
            test_point = PathSolver.get_point_in_direction(head, direction)
            if not game.is_collision(test_point):
                possible_directions.append(direction)
        
        if not possible_directions:
            return None  # No safe move
        
        # Choose direction that minimizes distance to food
        best_direction = None
        best_distance = float('inf')
        
        for direction in possible_directions:
            test_point = PathSolver.get_point_in_direction(head, direction)
            distance = abs(test_point.x - food.x) + abs(test_point.y - food.y)
            if distance < best_distance:
                best_distance = distance
                best_direction = direction
        
        return best_direction

    @staticmethod
    def get_point_in_direction(point, direction):
        """Get the point in the given direction"""
        if direction == Direction.RIGHT:
            return Point(point.x + BLOCK_SIZE, point.y)
        elif direction == Direction.LEFT:
            return Point(point.x - BLOCK_SIZE, point.y)
        elif direction == Direction.UP:
            return Point(point.x, point.y - BLOCK_SIZE)
        elif direction == Direction.DOWN:
            return Point(point.x, point.y + BLOCK_SIZE)
        return point

    @staticmethod
    def get_next_move_from_path(game, path):
        """Determine the next direction to move based on the path."""
        if not path or len(path) < 1:
            return None
        
        head = game.snake[0]
        current_grid_pos = (head.x // BLOCK_SIZE, head.y // BLOCK_SIZE)
        next_grid_pos = path[0]  # The next position in the path
        
        # Calculate the direction from current position to next position
        dx = next_grid_pos[0] - current_grid_pos[0]
        dy = next_grid_pos[1] - current_grid_pos[1]
        
        if dx == 1:
            return Direction.RIGHT
        elif dx == -1:
            return Direction.LEFT
        elif dy == 1:
            return Direction.DOWN
        elif dy == -1:
            return Direction.UP
        
        return None

    @staticmethod
    def hamiltonian_cycle_move(game):
        """Follow a Hamiltonian cycle pattern"""
        head = game.snake[0]
        grid_x = head.x // BLOCK_SIZE
        grid_y = head.y // BLOCK_SIZE
        
        # Simple Hamiltonian-like pattern (snake pattern)
        if grid_y % 2 == 0:  # Even rows go right
            if grid_x < (game.w // BLOCK_SIZE) - 1:
                return Direction.RIGHT
            else:
                return Direction.DOWN
        else:  # Odd rows go left
            if grid_x > 0:
                return Direction.LEFT
            else:
                return Direction.DOWN
        
        # If at bottom, go up on the sides
        if grid_y == (game.h // BLOCK_SIZE) - 1:
            if grid_x == 0:
                return Direction.RIGHT
            else:
                return Direction.LEFT
                
        return None

    @staticmethod
    def wall_follower(game):
        """Wall follower algorithm (right-hand rule)"""
        head = game.snake[0]
        current_dir = game.direction
        
        # Try directions in order: right, forward, left, backward
        directions_to_try = [
            PathSolver.rotate_right(current_dir),
            current_dir,
            PathSolver.rotate_left(current_dir),
            PathSolver.rotate_right(PathSolver.rotate_right(current_dir))
        ]
        
        for direction in directions_to_try:
            test_point = PathSolver.get_point_in_direction(head, direction)
            if not game.is_collision(test_point):
                return direction
                
        return current_dir  # Fallback

    @staticmethod
    def rotate_right(direction):
        """Rotate direction 90 degrees right"""
        if direction == Direction.RIGHT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.LEFT
        elif direction == Direction.LEFT:
            return Direction.UP
        elif direction == Direction.UP:
            return Direction.RIGHT

    @staticmethod
    def rotate_left(direction):
        """Rotate direction 90 degrees left"""
        if direction == Direction.RIGHT:
            return Direction.UP
        elif direction == Direction.UP:
            return Direction.LEFT
        elif direction == Direction.LEFT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.RIGHT

class AdvancedAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Training statistics
        self.total_reward = 0
        self.best_score = 0
        self.algorithm_mode = "hybrid"  # hybrid, dqn_only, pathfinder_only
        self.current_algorithm = "dqn"
        self.algorithm_performance = {"dqn": 0, "a_star": 0, "greedy": 0, "hamiltonian": 0, "wall_follower": 0}
        
        # Try to load existing model and training data
        self.load_model_and_data()

    def load_model_and_data(self):
        # Load model weights
        model_loaded = self.model.load()
        
        # Load training data if model was loaded
        if model_loaded:
            try:
                with open('./model/training_data.json', 'r') as f:
                    data = json.load(f)
                    self.n_games = data['n_games']
                    self.best_score = data.get('best_score', 0)
                    self.epsilon = max(5, 80 - self.n_games)
                    self.algorithm_mode = data.get('algorithm_mode', 'hybrid')
                    print(f"Resuming from game {self.n_games}, Best Score: {self.best_score}")
            except FileNotFoundError:
                print("No training data found, starting fresh")
            except Exception as e:
                print(f"Error loading training data: {e}")

    def save_training_data(self):
        data = {
            'n_games': self.n_games,
            'best_score': self.best_score,
            'epsilon': self.epsilon,
            'algorithm_mode': self.algorithm_mode,
            'algorithm_performance': self.algorithm_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        with open('./model/training_data.json', 'w') as f:
            json.dump(data, f, indent=4)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.total_reward += reward

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_dqn_action(self, state):
        """Get action from DQN model"""
        self.epsilon = max(5, 80 - self.n_games)
        
        # Exploration: random action
        if random.randint(0, 200) < self.epsilon:
            final_move = [0, 0, 0]
            move = random.randint(0, 2)
            final_move[move] = 1
            return final_move
        
        # Exploitation: use the model
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def get_algorithm_action(self, game, algorithm_name):
        """Get action from specified algorithm"""
        try:
            if algorithm_name == "a_star":
                path = PathSolver.shortest_path_a_star(game)
                if path:
                    direction = PathSolver.get_next_move_from_path(game, path)
                    return self.direction_to_action(direction, game.direction)
            
            elif algorithm_name == "greedy":
                direction = PathSolver.greedy_direction(game)
                if direction:
                    return self.direction_to_action(direction, game.direction)
            
            elif algorithm_name == "hamiltonian":
                direction = PathSolver.hamiltonian_cycle_move(game)
                if direction:
                    return self.direction_to_action(direction, game.direction)
            
            elif algorithm_name == "wall_follower":
                direction = PathSolver.wall_follower(game)
                if direction:
                    return self.direction_to_action(direction, game.direction)
            
            elif algorithm_name == "longest_path":
                path = PathSolver.longest_path(game)
                if path:
                    direction = PathSolver.get_next_move_from_path(game, path)
                    return self.direction_to_action(direction, game.direction)
                    
        except Exception as e:
            print(f"Error in {algorithm_name}: {e}")
        
        # Fallback to random if algorithm fails
        return self.get_dqn_action(self.get_state(game))

    def direction_to_action(self, direction, current_direction):
        """Convert direction to action vector"""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(current_direction)
        target_idx = clock_wise.index(direction)
        
        if target_idx == idx:
            return [1, 0, 0]  # straight
        elif target_idx == (idx + 1) % 4:
            return [0, 1, 0]  # right
        else:
            return [0, 0, 1]  # left

    def select_best_algorithm(self, game):
        """Dynamically select the best algorithm based on performance"""
        if self.n_games < 20:  # Early exploration phase
            algorithms = ["a_star", "greedy", "wall_follower"]
            return random.choice(algorithms)
        
        # Use the best performing algorithm recently
        best_algorithm = max(self.algorithm_performance, key=self.algorithm_performance.get)
        
        # Occasionally explore other algorithms
        if random.random() < 0.2:
            algorithms = list(self.algorithm_performance.keys())
            return random.choice(algorithms)
            
        return best_algorithm

    def update_algorithm_performance(self, algorithm_name, score):
        """Update algorithm performance metrics"""
        if algorithm_name in self.algorithm_performance:
            # Weighted average with decay
            self.algorithm_performance[algorithm_name] = (
                0.9 * self.algorithm_performance[algorithm_name] + 0.1 * score
            )

    def get_hybrid_action(self, game, state):
        """Hybrid approach combining multiple algorithms"""
        # Early games: use pathfinding more
        if self.n_games < 50 or random.random() < self.epsilon / 100:
            algorithm = self.select_best_algorithm(game)
            self.current_algorithm = algorithm
            action = self.get_algorithm_action(game, algorithm)
            return action
        
        # Later games: use DQN more
        self.current_algorithm = "dqn"
        return self.get_dqn_action(state)

    def get_action(self, game, state):
        """Main method to get action"""
        try:
            if self.algorithm_mode == "dqn_only":
                self.current_algorithm = "dqn"
                return self.get_dqn_action(state)
                
            elif self.algorithm_mode == "pathfinder_only":
                algorithm = self.select_best_algorithm(game)
                self.current_algorithm = algorithm
                return self.get_algorithm_action(game, algorithm)
                
            elif self.algorithm_mode == "hybrid":
                return self.get_hybrid_action(game, state)
                
            else:
                self.current_algorithm = "dqn"
                return self.get_dqn_action(state)
                
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Fallback to random action
            final_move = [0, 0, 0]
            final_move[random.randint(0, 2)] = 1
            return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = AdvancedAgent()
    game = SnakeGameAI()
    
    # Create directories
    for directory in ['./model', './plots', './results']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    print("Available algorithm modes: dqn_only, pathfinder_only, hybrid")
    print(f"Current mode: {agent.algorithm_mode}")
    
    # Training loop
    while True:
        try:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(game, state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # Update algorithm performance
                agent.update_algorithm_performance(agent.current_algorithm, score)

                if score > record:
                    record = score
                    agent.best_score = record
                    agent.model.save()
                    agent.save_training_data()

                print(f'Game {agent.n_games}, Score: {score}, Record: {record}, '
                      f'Algorithm: {agent.current_algorithm}, Epsilon: {agent.epsilon:.1f}')

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                # Plot every 10 games
                if agent.n_games % 10 == 0:
                    plot(plot_scores, plot_mean_scores)
                
                # Save training data every 10 games
                if agent.n_games % 10 == 0:
                    agent.save_training_data()
                    
                    # Save algorithm performance
                    with open('./results/algorithm_performance.csv', 'a') as f:
                        if agent.n_games == 10:
                            f.write('game,dqn,a_star,greedy,hamiltonian,wall_follower\n')
                        f.write(f"{agent.n_games},{agent.algorithm_performance['dqn']:.2f},")
                        f.write(f"{agent.algorithm_performance['a_star']:.2f},")
                        f.write(f"{agent.algorithm_performance['greedy']:.2f},")
                        f.write(f"{agent.algorithm_performance['hamiltonian']:.2f},")
                        f.write(f"{agent.algorithm_performance['wall_follower']:.2f}\n")
        
        except Exception as e:
            print(f"Error in training loop: {e}")
            game.reset()
            agent.n_games += 1


if __name__ == '__main__':
    train()