import random
import matplotlib.pyplot as plt
import heapq

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = None
        self.end = None

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

    def visualize(self):
        plt.figure(figsize=(8, 8))
        cmap = plt.get_cmap('tab20c', 4)  # 4种不同颜色：空白，障碍物，起点，终点
        plt.imshow(self.grid, cmap=cmap)
        plt.grid(True)
        plt.show()

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
        start = self.start
        end = self.end

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.neighbors(current):
                if not self.is_walkable(neighbor):
                    continue

                tentative_g_score = g_score[current] + 1  # 假设每移动一步代价为1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # 无路径可达

    def heuristic(self, a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x1 - x2) + abs(y1 - y2)  # 使用曼哈顿距离作为启发式函数

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

        plt.figure(figsize=(8, 8))
        cmap = plt.get_cmap('tab20c', 5)  # 5种颜色：空白，障碍物，起点，终点，路径
        plt.imshow(self.grid, cmap=cmap)
        plt.grid(True)
        plt.show()

# 使用示例
grid = Grid(10, 10)
grid.set_start(0, 0)
grid.set_end(9, 9)
grid.generate_obstacles(20)
grid.visualize()

# 运行A*算法并可视化路径
path = grid.astar()
if path:
    print("找到路径:", path)
    grid.visualize_path(path)
else:
    print("无可达路径")
