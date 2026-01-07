import numpy as np
import cv2
import os
import random
from tqdm import tqdm

# ================= 配置参数 =================
TRAIN_DIR = "train_images" 
GRID_SIZE = 20
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右

# --- 关键修改 1: 参数调整 ---
ALPHA = 0.1                 # 学习率
GAMMA = 0.95                # 折扣因子 (看重更长远的未来)
EPOCHS = 3000               # 训练轮数 (增加训练量)

# Epsilon 衰减参数
EPSILON_START = 1.0         # 一开始完全随机探索
EPSILON_END = 0.01          # 最后只保留 1% 的随机性
EPSILON_DECAY = 0.999       # 衰减速率

# ================= 环境类 =================
class TrainingEnv:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.defect_map = None

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        self.defect_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        h, w = gray.shape
        step_h = h // self.grid_size
        step_w = w // self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                row_start, row_end = i*step_h, (i+1)*step_h
                col_start, col_end = j*step_w, (j+1)*step_w
                block = gray[row_start:row_end, col_start:col_end]
                if block.size == 0: continue
                
                # Max Pooling 保持一致
                max_val = np.max(block)
                if max_val > 160: 
                    self.defect_map[i, j] = 1
        return True

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return False
        return self.preprocess_image(img)

    def get_reward(self, x, y, visited):
        # --- 关键修改 2: 奖励函数微调 ---
        
        # 1. 越界惩罚 (撞墙)
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -20 
        
        # 2. 回头路/重复访问惩罚 (加大力度，防止死循环)
        if visited[x, y]:
            return -5   
        
        # 3. 发现缺陷奖励 (给予巨大奖励，吸引它过去)
        if self.defect_map[x, y] == 1:
            return 100   
        
        # 4. 普通移动成本 (稍微给点压力，让它别磨蹭)
        return -0.5     

# ================= 训练主逻辑 =================
def train():
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    
    if not os.path.exists(TRAIN_DIR):
        print(f"错误：找不到 {TRAIN_DIR}")
        return

    image_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.bmp', '.png'))]
    if len(image_files) == 0:
        print("无训练图片")
        return

    print(f"开始训练，共 {len(image_files)} 张图，迭代 {EPOCHS} 轮...")
    
    env = TrainingEnv(GRID_SIZE)
    epsilon = EPSILON_START
    
    # 进度条
    pbar = tqdm(range(EPOCHS))
    
    for epoch in pbar:
        random.shuffle(image_files)
        
        # 每 10 张图衰减一次 Epsilon，保证足够的探索
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
            
        # 更新进度条显示的 Epsilon
        if epoch % 100 == 0:
            pbar.set_description(f"Training (Epsilon: {epsilon:.3f})")
        
        for img_path in image_files:
            if not env.load_image(img_path): continue
            
            x, y = 0, 0
            visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            visited[0, 0] = True
            
            # 增加单局步数限制，让它有机会走远点
            for _ in range(GRID_SIZE * GRID_SIZE): 
                # Epsilon-Greedy
                if random.random() < epsilon:
                    action_idx = random.randint(0, 3)
                else:
                    action_idx = np.argmax(q_table[x, y])
                
                dx, dy = ACTIONS[action_idx]
                nx, ny = x + dx, y + dy
                
                # 虚拟执行，用于判断越界
                is_out = not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE)
                
                # 计算奖励
                # 注意：如果越界，nx, ny 还是原来的 x, y (或者你可以设计为撞墙停在原地)
                # 这里我们逻辑是：如果越界，状态不变，但给负奖励
                if is_out:
                    reward = -20
                    next_max_q = np.max(q_table[x, y]) # 撞墙了，下一状态还是当前
                else:
                    reward = env.get_reward(nx, ny, visited)
                    next_max_q = np.max(q_table[nx, ny])
                
                # Q-Learning 更新
                old_q = q_table[x, y, action_idx]
                new_q = old_q + ALPHA * (reward + GAMMA * next_max_q - old_q)
                q_table[x, y, action_idx] = new_q
                
                # 状态跳转 (只有没越界才走)
                if not is_out:
                    x, y = nx, ny
                    visited[x, y] = True
                    
                    # 如果发现了缺陷，可以给一个 flag，或者继续走
                    # 为了训练它找更多缺陷，我们让它继续走

    np.save("q_table.npy", q_table)
    print("训练完成！模型已保存。")

if __name__ == "__main__":
    train()