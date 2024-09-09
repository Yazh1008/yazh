import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
import heapq
import time

class AStarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A* Algorithm Visualizer")
        
        # 参数配置
        self.grid_size_label = ttk.Label(root, text="Grid Size:")
        self.grid_size_label.grid(row=0, column=0)
        self.grid_size_entry = ttk.Entry(root)
        self.grid_size_entry.grid(row=0, column=1)
        self.grid_size_entry.insert(0, '10')

        self.obstacles_label = ttk.Label(root, text="Obstacles:")
        self.obstacles_label.grid(row=1, column=0)
        self.obstacles_entry = ttk.Entry(root)
        self.obstacles_entry.grid(row=1, column=1)
        self.obstacles_entry.insert(0, '20')

        self.seed_label = ttk.Label(root, text="Random Seed:")
        self.seed_label.grid(row=2, column=0)
        self.seed_entry = ttk.Entry(root)
        self.seed_entry.grid(row=2, column=1)
        self.seed_entry.insert(0, '42')

        # 按钮：生成网格和运行A*
        self.generate_button = ttk.Button(root, text="Generate Grid", command=self.generate_grid)
        self.generate_button.grid(row=3, column=0)

        self.run_button = ttk.Button(root, text="Run A*", command=self.run_astar)
        self.run_button.grid(row=3, column=1)

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.grid(row=4, column=0, columnspan=2)

        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=5, column=0, columnspan=2)

        self.grid = None  # 初始化为空

    def generate_grid(self):
        grid_size = int(self.grid_size_entry.get())
        num_obstacles = int(self.obstacles_entry.get())
        seed = int(self.seed_entry.get())

        self.grid = Grid(grid_size, grid_size, seed)
        self.grid.set_start(0, 0)
        self.grid.set_end(grid_size-1, grid_size-1)
        self.grid.generate_obstacles(num_obstacles)
        self.visualize_grid(self.grid.grid)

    def visualize_grid(self, grid_data):
        self.canvas.delete("all")
        cell_width = 400 // len(grid_data)
        for y in range(len(grid_data)):
            for x in range(len(grid_data[0])):
                color = "white"
                if grid_data[y][x] == 1:
                    color = "black"
                elif grid_data[y][x] == 2:
                    color = "green"  # 起点
                elif grid_data[y][x] == 3:
                    color = "red"  # 终点
                elif grid_data[y][x] == 4:
                    color = "blue"  # 路径

                self.canvas.create_rectangle(x * cell_width, y * cell_width, 
                                             (x + 1) * cell_width, (y + 1) * cell_width, 
                                             fill=color, outline="gray")

    def run_astar(self):
        if self.grid:
            path, explored, path_blocks, ratio, time_taken = self.grid.astar()
            if path:
                self.grid.visualize_path(path)
                self.visualize_grid(self.grid.grid)
                self.result_label.config(text=f"Path Found! Time: {time_taken:.2f}s, Ratio: {ratio:.2f}")
            else:
                self.result_label.config(text="No Path Found.")

class Grid:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = None
        self.end = None
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def set_start(self, x, y):
        self.start = (x, y)
        self.grid[y][x] = 2  # 用2表示起点

    def set_end(self, x, y):
        self.end = (x, y)
        self.grid[y][x] = 3  # 用3表示终点

    def generate_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            while self.grid[y][x] != 0:  # 确保不会放置在起点/终点或其他障碍物上
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.grid[y][x] = 1  # 1代表一个障碍物

    def neighbors(self, node):
        x, y = node
        neighbors = []
        if x > 0: neighbors.append((x-1, y))  # 左
        if x < self.width - 1: neighbors.append((x+1, y))  # 右
        if y > 0: neighbors.append((x, y-1))  # 上
        if y < self.height - 1: neighbors.append((x, y+1))  # 下
        return neighbors

    def is_walkable(self, node):
        x, y = node
        return self.grid[y][x] == 0 or self.grid[y][x] == 3  # 0为空白区域，3为终点

    def astar(self):
        start_time = time.time()
        start = self.start
        end = self.end

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        explored = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            explored.add(current)

            if current == end:
                end_time = time.time()
                path = self.reconstruct_path(came_from, current)
                explored_blocks = len(explored)
                path_blocks = len(path)
                ratio = path_blocks / explored_blocks
                time_taken = end_time - start_time
                return path, explored_blocks, path_blocks, ratio, time_taken

            for neighbor in self.neighbors(current):
                if not self.is_walkable(neighbor):
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None, len(explored), 0, 0, time.time() - start_time  # 无路径时返回

    def heuristic(self, a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def visualize_path(self, path):
        for (x, y) in path:
            if self.grid[y][x] != 2 and self.grid[y][x] != 3:
                self.grid[y][x] = 4  # 4代表路径

if __name__ == "__main__":
    root = tk.Tk()
    app = AStarApp(root)
    root.mainloop()

