# 🏭 工业表面划伤缺陷智能检测路径规划系统
# (Industrial Surface Defect Detection & Path Planning System)

> 🎓 **人工智能原理 (Artificial Intelligence Principles)** 课程期末大作业  
> 👨‍💻 设计与实现：**mmh**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Blog](https://img.shields.io/badge/Blog-zakee.fun-orange.svg)](https://zakee.fun)

## 📖 项目简介 (Introduction)

本项目是一个可视化的工业缺陷检测路径规划仿真系统。系统旨在通过对比传统算法与强化学习算法在工业表面扫描场景下的表现，探索更高效的缺陷检测策略。

项目重点模拟了智能体（Agent）在未知环境下的决策过程，提供了直观的 Web 交互界面，支持多种算法的实时对比与数据统计。

🔗 **在线演示 (Live Demo):** [https://neu.zakee.fun](https://neu.zakee.fun)

## 📂 数据集 (Dataset)

本项目的基础环境构建与测试基于 **NEU Surface Defect Database**。该数据集收集了热轧钢带的六种典型表面缺陷（如划痕、斑块、氧化铁皮等），是工业表面缺陷检测领域的经典基准数据集。

🔗 **数据来源 (Source):** [Kaggle - NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data)

## 🧠 核心算法 (Core Algorithms)

本系统实现了四种不同机制的路径规划算法，并在同一环境下进行“背对背”性能对比：

1.  **传统全覆盖扫描 (Traditional Coverage)**
    * 基于“弓”字形的遍历策略，模拟传统的自动化扫描逻辑，作为性能基线 (Baseline)。
2.  **启发式抽样 (Heuristic Sampling)**
    * 引入基于概率的启发式规则，在探索与利用之间寻找简单的平衡。
3.  **表格型 Q-Learning (Tabular Q-Learning)**
    * 经典的无模型强化学习算法，通过维护 Q-Table 学习状态-动作价值，实现自主避障与目标搜寻。
4.  **深度强化学习 (Deep Q-Network, DQN)**
    * 结合深度神经网络与 Q-Learning，利用经验回放 (Experience Replay) 解决高维状态空间下的复杂决策问题。

## 🛠️ 技术栈 (Tech Stack)

* **算法核心**: Python, NumPy, PyTorch (DQN Implementation)
* **Web 框架**: Gradio / Flask (Backend), HTML/JS (Frontend Integration)
* **部署**: Hugging Face Spaces + GitHub Pages (Custom Domain)

## 🚀 快速开始 (Quick Start)

如果你希望在本地运行本项目：

```bash
# 1. 克隆仓库
git clone [https://github.com/momo325/你的仓库名.git](https://github.com/momo325/你的仓库名.git)

# 2. 进入目录
cd 你的仓库名

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
python app.py
