import os
# --- 1. 解决 OpenMP 冲突 (必须放在最前面) ---
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.environ['FONTCONFIG_PATH'] = '/tmp/fontconfig_cache'

import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ================= 配置参数 =================
GRID_SIZE = 20
TOTAL_CELLS = GRID_SIZE * GRID_SIZE 
VIEW_SIZE = 5               
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

# ================= 1. 定义神经网络结构 =================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ================= 2. 加载模型 =================
Q_TABLE = None
DQN_MODEL = None

def load_models():
    global Q_TABLE, DQN_MODEL
    info_text = ""
    
    if os.path.exists("q_table.npy"):
        Q_TABLE = np.load("q_table.npy")
        info_text += "✅ **Q-Learning**\n- 状态: 20x20\n\n"
    else:
        Q_TABLE = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        info_text += "❌ **Q-Learning**: 未加载\n\n"

    if os.path.exists("dqn_model.pth"):
        try:
            input_dim = VIEW_SIZE * VIEW_SIZE
            output_dim = 4
            model = DQN(input_dim, output_dim)
            model.load_state_dict(torch.load("dqn_model.pth", map_location='cpu'))
            model.eval()
            DQN_MODEL = model
            info_text += "✅ **DQN Agent**\n- 输入: 视觉+记忆"
        except Exception as e:
            info_text += f"❌ DQN 错误: {str(e)[:10]}"
    else:
        info_text += "❌ **DQN**: 未找到"
        
    return info_text

MODEL_INFO_TEXT = load_models()

# ================= 可视化与环境类 (保持不变) =================
def plot_inspection(env, path):
    h, w = env.grid_size, env.grid_size
    img_h, img_w = env.original_img.shape[:2]
    scanned_mask = np.zeros((h, w))
    for r, c in path: scanned_mask[r, c] = 1
    defect_mask = env.defect_map
    
    detected_mask = (defect_mask == 1) & (scanned_mask == 1)
    missed_mask = (defect_mask == 1) & (scanned_mask == 0)
    safe_mask = (defect_mask == 0) & (scanned_mask == 1)
    
    dpi = 100
    fig, ax = plt.subplots(figsize=(img_w/dpi, img_h/dpi), dpi=dpi)
    if len(env.original_img.shape) == 3:
        gray_bg = cv2.cvtColor(env.original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_bg = env.original_img
    ax.imshow(gray_bg, cmap='gray', alpha=0.6)
    
    if np.sum(missed_mask) > 0:
        missed_layer = np.zeros((h, w, 4))
        missed_layer[missed_mask] = [1, 0.8, 0, 0.6] 
        ax.imshow(missed_layer, extent=[0, img_w, img_h, 0], interpolation='nearest')
    if np.sum(safe_mask) > 0:
        green_layer = np.zeros((h, w, 4))
        green_layer[safe_mask] = [0, 1, 0, 0.4] 
        ax.imshow(green_layer, extent=[0, img_w, img_h, 0], interpolation='nearest')
    if np.sum(detected_mask) > 0:
        red_layer = np.zeros((h, w, 4))
        red_layer[detected_mask] = [1, 0, 0, 0.8] 
        ax.imshow(red_layer, extent=[0, img_w, img_h, 0], interpolation='nearest')

    if len(path) > 1:
        step_y = img_h / h; step_x = img_w / w
        ys = [r * step_y + step_y/2 for r, c in path]
        xs = [c * step_x + step_x/2 for r, c in path]
        ax.plot(xs, ys, color='darkblue', linewidth=2, alpha=0.9)
        ax.scatter(xs[0], ys[0], c='white', edgecolors='black', s=60, zorder=10)
        ax.scatter(xs[-1], ys[-1], c='red', marker='x', s=60, zorder=10)

    plt.axis('off'); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0,0)
    fig.canvas.draw()
    data = np.array(fig.canvas.buffer_rgba()); plt.close(fig)
    return data[:, :, :3]

class DefectEnv:
    def __init__(self, image):
        self.grid_size = GRID_SIZE
        self.original_img = image 
        self.defect_map = None; self.raw_gray = None 
        self._preprocess(image) 
    def _preprocess(self, image):
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image
        self.raw_gray = cv2.resize(gray, (self.grid_size, self.grid_size))
        self.defect_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        h, w = gray.shape; step_h = h // self.grid_size; step_w = w // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                row_start, row_end = i*step_h, (i+1)*step_h
                col_start, col_end = j*step_w, (j+1)*step_w
                block = gray[row_start:row_end, col_start:col_end]
                if block.size == 0: continue
                if np.max(block) > 160: self.defect_map[i, j] = 1
    def get_state(self, x, y, visited):
        pad = VIEW_SIZE // 2
        padded_img = np.pad(self.raw_gray, pad_width=pad, mode='constant', constant_values=0)
        view_img = padded_img[x:x+VIEW_SIZE, y:y+VIEW_SIZE] / 255.0
        visited_int = visited.astype(int)
        padded_visit = np.pad(visited_int, pad_width=pad, mode='constant', constant_values=1)
        view_visit = padded_visit[x:x+VIEW_SIZE, y:y+VIEW_SIZE]
        return np.concatenate((view_img.flatten(), view_visit.flatten()))
    def check(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size: return self.defect_map[x, y] == 1
        return False

# ================= 算法逻辑 (精简版) =================
def method_baseline(env):
    path = []; scan_count = 0; detected = 0
    for i in range(env.grid_size):
        col_range = range(env.grid_size) if i % 2 == 0 else range(env.grid_size-1, -1, -1)
        for j in col_range:
            path.append((i, j)); scan_count += 1
            if env.check(i, j): detected += 1
    return path, scan_count, detected

def method_heuristic(env):
    path = []; scan_count = 0; detected = 0
    visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    queue = []
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i + j) % 2 == 0: queue.append((i, j))
    while queue:
        x, y = queue.pop(0)
        if visited[x, y]: continue
        visited[x, y] = True; path.append((x, y)); scan_count += 1
        if env.check(x, y): 
            detected += 1
            neighbors = [(x+dx, y+dy) for dx in [-1,0,1] for dy in [-1,0,1] if not (dx==0 and dy==0)]
            for nx, ny in neighbors:
                if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size and not visited[nx, ny]:
                    queue.insert(0, (nx, ny))
    return path, scan_count, detected

def method_q_table(env):
    path = []; scan_count = 0; detected = 0
    visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    curr = (0, 0); path.append(curr); visited[0, 0] = True; path_set = {curr} 
    max_steps = env.grid_size * env.grid_size 
    for _ in range(max_steps):
        x, y = curr
        action_idx = np.argmax(Q_TABLE[x, y])
        dx, dy = ACTIONS[action_idx]
        nx, ny = x + dx, y + dy
        nx = max(0, min(env.grid_size-1, nx)); ny = max(0, min(env.grid_size-1, ny))
        if (nx, ny) == curr or (nx, ny) in path_set: break 
        curr = (nx, ny); path.append(curr); path_set.add(curr); scan_count += 1
        if not visited[nx, ny]:
            visited[nx, ny] = True
            if env.check(nx, ny): detected += 1
    return path, scan_count, detected

def method_dqn(env):
    if DQN_MODEL is None: return [], 0, 0 
    path = []; scan_count = 0; detected = 0
    visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    curr = (0, 0); path.append(curr); visited[curr] = True
    max_steps = env.grid_size * env.grid_size * 1.5; stuck_counter = 0 
    with torch.no_grad():
        for _ in range(int(max_steps)):
            x, y = curr
            state = env.get_state(x, y, visited)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_idx = DQN_MODEL(state_tensor).argmax().item()
            dx, dy = ACTIONS[action_idx]
            nx, ny = max(0, min(env.grid_size-1, x+dx)), max(0, min(env.grid_size-1, y+dy))
            if (nx, ny) == curr: stuck_counter += 1
            else: stuck_counter = 0 
            if stuck_counter > 5: break
            curr = (nx, ny); path.append(curr)
            if not visited[nx, ny]:
                visited[nx, ny] = True; scan_count += 1
                if env.check(nx, ny): detected += 1
    return path, scan_count, detected

# ================= 界面逻辑 =================
def run_demo(image):
    if image is None: return [None]*4 + [""]*4
    env = DefectEnv(image)
    
    p1, c1, d1 = method_baseline(env)
    p2, c2, d2 = method_heuristic(env)
    p3, c3, d3 = method_q_table(env)
    p4, c4, d4 = method_dqn(env)
    
    img1 = plot_inspection(env, p1)
    img2 = plot_inspection(env, p2)
    img3 = plot_inspection(env, p3)
    img4 = plot_inspection(env, p4)
    
    total = np.sum(env.defect_map)
    def fmt(steps, detected):
        det_rate = (detected / total * 100) if total > 0 else (100 if detected==0 else 0)
        cov_rate = (steps / TOTAL_CELLS) * 100
        return f"总步数: {steps}\n检出: {detected}/{total}\n检出率: {det_rate:.1f}%\n覆盖率: {cov_rate:.1f}%"
    
    return (
        img1, fmt(c1, d1),
        img2, fmt(c2, d2),
        img3, fmt(c3, d3),
        img4, fmt(c4, d4)
    )

# ================= 启动 Gradio =================
custom_css = """
.stat-box textarea { min-height: 110px !important; font-size: 14px !important; font-family: 'Consolas', monospace; }
.info-box { background: #f3f4f6; padding: 12px; border-radius: 8px; font-size: 13px; }
.legend-box { margin-top: 5px; padding: 8px; border: 1px solid #eee; border-radius: 5px; font-size: 13px; }
"""

with gr.Blocks( title="智能表面划伤缺陷检测系统 Pro") as demo:
    gr.Markdown("## 工业表面划伤缺陷智能检测路径规划系统 (AI Lab)")
    
    # --- 上半部分：控制区 (左信息 + 右上传) ---
    with gr.Row():
        # 左侧：系统情报 (Scale 1)
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ 系统状态")
            gr.Markdown(MODEL_INFO_TEXT, elem_classes="info-box")
            
        # 右侧：上传与按钮 (Scale 3)
        with gr.Column(scale=3):
            with gr.Row():
                # 图片上传
                with gr.Column(scale=3):
                    inp = gr.Image(label="工件影像输入 (NEU Dataset)", height=240)
                
                # 按钮与图例
                with gr.Column(scale=1):
                    btn = gr.Button("🚀 全模式对比扫描", variant="primary", size="lg")
                    gr.HTML("""
                    <div class="legend-box">
                        <b>ℹ️ 图例</b><br>
                        <span style="color:red;">●</span> 检出<br> <span style="color:#eab308;">●</span> 漏检<br>
                        <span style="color:green;">●</span> 安全<br> <span style="color:blue;">●</span> 路径
                    </div>
                    """)

    # --- 下半部分：结果区 (独占一行，四列排开) ---
    gr.Markdown("---")
    gr.Markdown("### 📊 扫描结果对比")
    
    # 这里不再嵌套在右侧 Column 里，而是全新的 Row，从而保证全宽
    with gr.Row():
        with gr.Column():
            gr.Markdown("**1. 传统全覆盖**")
            o1_img = gr.Image(show_label=False)
            o1_txt = gr.Textbox(label="统计数据", elem_classes="stat-box")
        with gr.Column():
            gr.Markdown("**2. 启发式抽样**")
            o2_img = gr.Image(show_label=False)
            o2_txt = gr.Textbox(label="统计数据", elem_classes="stat-box")
        with gr.Column():
            gr.Markdown("**3. 表格型 Q-Learning**")
            o3_img = gr.Image(show_label=False)
            o3_txt = gr.Textbox(label="统计数据", elem_classes="stat-box")
        with gr.Column():
            gr.Markdown("**4. 深度强化学习 (DQN)**")
            o4_img = gr.Image(show_label=False)
            o4_txt = gr.Textbox(label="统计数据", elem_classes="stat-box")
            
    btn.click(run_demo, inputs=inp, outputs=[o1_img, o1_txt, o2_img, o2_txt, o3_img, o3_txt, o4_img, o4_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,css=custom_css)