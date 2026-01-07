import numpy as np
import cv2
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# 解决 OpenMP 冲突
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置参数 =================
TRAIN_DIR = "train_images"
GRID_SIZE = 20
VIEW_SIZE = 5               # 视野大小
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右
BATCH_SIZE = 128            # 增大 Batch Size
GAMMA = 0.95                
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995      # 让探索衰减得慢一点
TARGET_UPDATE = 20          
MEMORY_SIZE = 20000         
LR = 0.0005                 # 降低学习率，防止震荡
EPOCHS = 5000               # 5000次高质量训练足矣

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# ================= 1. 深度神经网络 (双通道输入) =================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 输入维度变成了 50 (25像素 + 25记忆)
        self.fc1 = nn.Linear(input_size * 2, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ================= 2. 环境类 =================
class TrainingEnv:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.defect_map = None
        self.raw_gray = None 

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Resize
        self.raw_gray = cv2.resize(gray, (self.grid_size, self.grid_size))
        # 二值化生成缺陷图
        self.defect_map = np.zeros_like(self.raw_gray)
        self.defect_map[self.raw_gray > 160] = 1
        return True

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return False
        return self.preprocess_image(img)

    def get_state(self, x, y, visited):
        """
        核心修改：同时获取【视觉信息】和【记忆信息】
        """
        pad = VIEW_SIZE // 2
        
        # 1. 视觉层 (归一化)
        padded_img = np.pad(self.raw_gray, pad_width=pad, mode='constant', constant_values=0)
        view_img = padded_img[x:x+VIEW_SIZE, y:y+VIEW_SIZE] / 255.0
        
        # 2. 记忆层 (0/1) - 让AI知道哪里走过了
        visited_int = visited.astype(int)
        padded_visit = np.pad(visited_int, pad_width=pad, mode='constant', constant_values=1) # 边界视为已访问
        view_visit = padded_visit[x:x+VIEW_SIZE, y:y+VIEW_SIZE]
        
        # 3. 拼接 (Flatten后拼接) -> 长度 50
        state = np.concatenate((view_img.flatten(), view_visit.flatten()))
        return state

    def step(self, x, y, action_idx, visited):
        dx, dy = ACTIONS[action_idx]
        nx, ny = x + dx, y + dy
        
        # 撞墙惩罚
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return x, y, -5, True 
        
        # 默认移动惩罚 (鼓励最高效)
        reward = -0.1
        
        if self.defect_map[nx, ny] == 1:
            if not visited[nx, ny]:
                reward = 10.0  # 发现新大陆！
            else:
                reward = -2.0  # 笨蛋，这里有缺陷但是你来过了，快走！
        else:
            if visited[nx, ny]:
                reward = -1.0  # 走回头路，且是空地，惩罚
            
        return nx, ny, reward, False

# ================= 3. 训练主逻辑 =================
def train():
    input_dim = VIEW_SIZE * VIEW_SIZE # 25
    output_dim = 4 
    
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    
    if not os.path.exists(TRAIN_DIR):
        print(f"请创建 {TRAIN_DIR}")
        return
    image_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(('.jpg','.png'))]
    
    env = TrainingEnv(GRID_SIZE)
    epsilon = EPSILON_START
    
    pbar = tqdm(range(EPOCHS))
    
    for epoch in pbar:
        img_path = random.choice(image_files)
        if not env.load_image(img_path): continue
        
        # 随机出生点
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        visited[x, y] = True
        
        # 获取初始状态 (双通道)
        state = env.get_state(x, y, visited)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        total_reward = 0
        
        for _ in range(GRID_SIZE * 3): # 步数限制
            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action_idx = q_values.argmax().item()
            
            nx, ny, reward, hit_wall = env.step(x, y, action_idx, visited)
            
            # 只有没撞墙才更新visited
            if not hit_wall:
                visited[nx, ny] = True
            
            next_state = env.get_state(nx, ny, visited)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            # 存储记忆
            memory.append((state_tensor, action_idx, reward, next_state_tensor))
            
            # 经验回放
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_state = torch.cat([x[0] for x in batch])
                b_action = torch.tensor([x[1] for x in batch]).unsqueeze(1).to(device)
                b_reward = torch.tensor([x[2] for x in batch]).unsqueeze(1).to(device)
                b_next_state = torch.cat([x[3] for x in batch])
                
                current_q = policy_net(b_state).gather(1, b_action)
                with torch.no_grad():
                    max_next_q = target_net(b_next_state).max(1)[0].unsqueeze(1)
                    target_q = b_reward + GAMMA * max_next_q
                
                loss = nn.SmoothL1Loss()(current_q, target_q) # Hubber Loss 更稳定
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            state_tensor = next_state_tensor
            x, y = nx, ny
            total_reward += reward
        
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
            
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        pbar.set_description(f"Eps:{epsilon:.2f} | R:{total_reward:.1f}")

    torch.save(policy_net.state_dict(), "dqn_model.pth")
    print("🎉 双通道 DQN 模型训练完成！")

if __name__ == "__main__":
    train()