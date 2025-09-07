# 🐍 Snake AI Project

Welcome to the **Snake AI** project! This is a Python-based implementation of the classic Snake game, enhanced with artificial intelligence. The snake is controlled by a combination of **reinforcement learning (Deep Q-Learning)** and **pathfinding algorithms** (A*, greedy, Hamiltonian cycle, wall follower), making it capable of learning to navigate, eat food, and avoid collisions intelligently. The project supports three modes: `dqn_only` (pure RL), `pathfinder_only` (algorithm-based), and `hybrid` (combining both).

This README provides everything you need to set up, run, and understand the project. Whether you're a beginner or an AI enthusiast, you can explore how AI powers this classic game! 🚀

## 📖 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Training and Visualization](#training-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features
- **Classic Snake Game**: Built with Pygame, where the snake eats red food, grows, and avoids walls/self-collisions.
- **Reinforcement Learning (DQN)**: A neural network (PyTorch) learns optimal moves through trial and error.
- **Pathfinding Algorithms**:
  - **A***: Finds the shortest path to food.
  - **Longest Path**: Extends A* with safe detours to avoid traps.
  - **Greedy**: Moves toward food while avoiding obstacles.
  - **Hamiltonian Cycle**: Follows a zigzag pattern to cover the grid.
  - **Wall Follower**: Sticks to walls for safer navigation.
- **Three AI Modes**:
  - `dqn_only`: Uses the neural network for decisions.
  - `pathfinder_only`: Relies on pathfinding algorithms.
  - `hybrid`: Combines RL and pathfinding for balanced performance.
- **Rewards System**:
  - +15 for eating food.
  - -10 for collisions (walls or self).
  - +1/-1 for moving closer/farther from food.
- **Path Visualization**: Press 'V' during gameplay to see the AI’s planned path (yellow/green lines).
- **Training Progress**: Plots scores and average scores to track improvement.
- **Save/Load**: Saves model weights (`model.pth`), training data (`training_data.json`), and algorithm performance (`algorithm_performance.csv`) on exit or every 10 games.
- **Error Handling**: Robust try-except blocks ensure stable training.

## 🛠️ Requirements
- Python 3.8+
- Libraries:
  - `pygame` (for game rendering)
  - `torch` (for neural network and RL)
  - `numpy` (for array operations)
  - `matplotlib` (for plotting training progress)
  - `ipython` (optional, for interactive plotting)

## 📦 Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/snake-ai.git
   cd snake-ai
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install pygame torch numpy matplotlib ipython
   ```

4. **Verify Setup**:
   Ensure all libraries are installed:
   ```bash
   python -c "import pygame, torch, numpy, matplotlib"
   ```

## 🚀 Usage
1. **Run the Game**:
   ```bash
   python agent.py
   ```
   This starts the training loop in the default `hybrid` mode. The game window (640x480) will appear, showing the snake in action.

2. **Interact with the Game**:
   - Press **'V'** to toggle path visualization (yellow/green lines show the AI’s planned path).
   - Close the window to save progress (model weights, training data, and performance metrics).

3. **Change AI Mode**:
   Modify the `algorithm_mode` in `agent.py` (line ~259) to one of:
   - `"dqn_only"`: Pure reinforcement learning.
   - `"pathfinder_only"`: Pathfinding algorithms only.
   - `"hybrid"`: Combines both (default).

4. **View Training Progress**:
   - Check the `plots/training_plot.png` file for a graph of scores and average scores.
   - Algorithm performance is saved in `results/algorithm_performance.csv`.

5. **Resume Training**:
   The project automatically loads `model.pth` and `training_data.json` from the `model/` folder if available, resuming from the last session.

## 📂 File Structure
```
snake-ai/
├── agent.py              # AI logic: combines DQN and pathfinding, manages training
├── game.py               # Game logic: Snake game rendering, movement, rewards
├── model.py              # Neural network: DQN model and training logic
├── helper.py             # Visualization: plots training scores
├── model/                # Stores model.pth and training_data.json
├── plots/                # Stores training_plot.png
├── results/              # Stores algorithm_performance.csv
└── README.md             # This file
```

### File Details
- **`game.py`**:
  - Defines `SnakeGameAI` class with game logic (movement, collisions, rewards).
  - Uses Pygame to render a 640x480 grid (20x20 pixel blocks).
  - Implements path visualization (press 'V') and displays score/algorithm.
- **`model.py`**:
  - Defines `Linear_QNet` (neural network: 11 inputs, 256 hidden, 3 outputs).
  - Implements `QTrainer` for Q-learning with Adam optimizer and MSE loss.
  - Handles model saving/loading (`model.pth`).
- **`agent.py`**:
  - Contains `PathSolver` with pathfinding algorithms (A*, longest path, etc.).
  - Defines `AdvancedAgent` for decision-making in three modes.
  - Manages training loop, experience replay, and performance tracking.
- **`helper.py`**:
  - Plots scores and average scores using Matplotlib.
  - Saves plots to `plots/training_plot.png`.

## 🧠 How It Works
The Snake AI combines two approaches:
1. **Reinforcement Learning (DQN)**:
   - The snake learns by trial and error, using a neural network to predict the best action (straight, right, left).
   - The state (11 inputs) includes danger ahead/right/left, current direction, and food position.
   - Rewards guide learning: +15 (food), -10 (collision), +1/-1 (closer/farther).
   - Experience replay stores past moves for batch training.
2. **Pathfinding Algorithms**:
   - **A***: Uses Manhattan distance to find the shortest path to food, avoiding obstacles.
   - **Longest Path**: Extends A* with random detours for safety.
   - **Greedy**: Picks the safest move closest to food.
   - **Hamiltonian Cycle**: Follows a zigzag pattern to cover the grid.
   - **Wall Follower**: Sticks to walls using a right-hand rule.
3. **Hybrid Mode**:
   - Early games (<50 or random chance): Uses pathfinding for stability.
   - Later: Shifts to DQN as the neural network improves.
   - Tracks algorithm performance to dynamically select the best approach.

The game runs at 40 FPS, with the snake moving in a 20x20 pixel grid. Progress is saved automatically, and plots show how the snake improves over time.

## 📈 Training and Visualization
- **Training**: Run `agent.py` to start. The snake plays games, learns, and saves progress every 10 games or on exit.
- **Visualization**:
  - Press 'V' to see the AI’s path (yellow lines, green start).
  - Check `plots/training_plot.png` for score trends.
  - View `results/algorithm_performance.csv` for algorithm comparison.
- **Tips**:
  - Run for 100+ games to see significant learning.
  - Tweak `epsilon` (exploration rate) or `gamma` (discount factor) in `agent.py` to adjust learning behavior.

## 🤝 Contributing
Want to improve this project? Here’s how:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-idea`).
3. Add your changes (e.g., new algorithms, UI improvements).
4. Test thoroughly with `python agent.py`.
5. Submit a pull request with a clear description.

Ideas for contributions:
- Add new pathfinding algorithms.
- Enhance the UI with sound effects or colors.
- Optimize the neural network (e.g., adjust layers).

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
