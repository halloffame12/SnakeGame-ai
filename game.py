import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI with Advanced Algorithms')
        self.clock = pygame.time.Clock()
        self.reset()
        self.current_algorithm = "dqn"
        self.show_path = False
        self.path = []

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.path = []

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def set_algorithm(self, algorithm_name):
        """Set the current algorithm for visualization"""
        self.current_algorithm = algorithm_name

    def set_path(self, path):
        """Set path for visualization"""
        self.path = path if path else []

    def play_step(self, action):
        self.frame_iteration += 1
        old_score = self.score
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:  # Toggle path visualization
                    self.show_path = not self.show_path
        
        # Store old head position for distance calculation
        old_head = self.head
        
        # Move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
            self.path = []  # Clear path when food is eaten
        else:
            self.snake.pop()
            
            # Give small reward/penalty based on movement toward food
            old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            
            if new_distance < old_distance:
                reward = 1  # Reward for moving toward food
            elif new_distance > old_distance:
                reward = -1  # Penalty for moving away
        
        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            
        # Hit boundary
        if (pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0):
            return True
            
        # Hit self
        if pt in self.snake[1:]:
            return True
            
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw path if enabled
        if self.show_path and self.path:
            for i, (x, y) in enumerate(self.path):
                # Draw path points
                color = GREEN if i == 0 else YELLOW  # Different color for start
                pygame.draw.rect(self.display, color, 
                               pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)
                
                # Draw path connections
                if i < len(self.path) - 1:
                    next_x, next_y = self.path[i + 1]
                    pygame.draw.line(self.display, PURPLE,
                                   (x*BLOCK_SIZE + BLOCK_SIZE//2, y*BLOCK_SIZE + BLOCK_SIZE//2),
                                   (next_x*BLOCK_SIZE + BLOCK_SIZE//2, next_y*BLOCK_SIZE + BLOCK_SIZE//2), 2)

        # Draw snake
        for i, pt in enumerate(self.snake):
            color = BLUE1 if i == 0 else BLUE2
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score and info
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        algo_text = font.render("Algorithm: " + self.current_algorithm, True, WHITE)
        self.display.blit(algo_text, [0, 30])
        
        path_text = font.render("Show Path (V): " + str(self.show_path), True, WHITE)
        self.display.blit(path_text, [0, 60])
        
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)