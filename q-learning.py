import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.table import Table

# 定义网格世界的大小
WORLD_SIZE = 10
# 定义起点和终点位置
START = (0, 0)
GOAL = (9, 9)

# 定义可能的动作: 上、下、左、右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['上', '下', '左', '右']

# 定义Q-learning的参数
EPSILON = 0.1  # 探索率
ALPHA = 0.5    # 学习率
GAMMA = 0.9    # 折扣因子
EPISODES = 1000  # 训练回合数

def step(state, action):
    """执行动作并返回新状态和奖励"""
    i, j = state
    di, dj = action
    
    # 计算新位置
    new_i = max(0, min(i + di, WORLD_SIZE - 1))
    new_j = max(0, min(j + dj, WORLD_SIZE - 1))
    
    # 如果撞到墙壁，保持原位
    if new_i == i and new_j == j:
        return (new_i, new_j), -1  # 撞墙惩罚
    
    # 到达终点奖励
    if (new_i, new_j) == GOAL:
        return (new_i, new_j), 10
    
    # 其他情况的小奖励
    return (new_i, new_j), -0.1

def choose_action(state, q_table):
    """基于ε-贪婪策略选择动作"""
    if random.random() < EPSILON:
        # 随机选择动作（探索）
        return random.choice(range(len(ACTIONS)))
    else:
        # 选择Q值最大的动作（利用）
        return np.argmax(q_table[state[0], state[1], :])

def print_policy(q_table):
    """打印学习到的策略"""
    policy = []
    for i in range(WORLD_SIZE):
        row = []
        for j in range(WORLD_SIZE):
            if (i, j) == GOAL:
                row.append('G')
                continue
            # 找到最佳动作
            best_action = np.argmax(q_table[i, j, :])
            row.append(ACTION_NAMES[best_action])
        policy.append(row)
    
    print("学习到的策略:")
    for row in policy:
        print(" ".join(row))

def plot_results(rewards, q_table):
    """可视化训练结果"""
    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('每回合奖励')
    plt.xlabel('回合数')
    plt.ylabel('奖励')
    
    # 绘制Q值热图
    plt.subplot(1, 2, 2)
    # 取每个状态的最大Q值
    max_q = np.max(q_table, axis=2)
    plt.imshow(max_q, cmap='hot', interpolation='nearest')
    plt.colorbar(label='最大Q值')
    plt.title('每个状态的最大Q值')
    plt.xticks(range(WORLD_SIZE))
    plt.yticks(range(WORLD_SIZE))
    
    # 标记起点和终点
    plt.text(START[1], START[0], 'S', ha='center', va='center', color='blue', fontsize=12)
    plt.text(GOAL[1], GOAL[0], 'G', ha='center', va='center', color='green', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def run_q_learning():
    """运行Q-learning算法"""
    # 初始化Q表
    q_table = np.zeros((WORLD_SIZE, WORLD_SIZE, len(ACTIONS)))
    rewards = []
    
    for episode in range(EPISODES):
        state = START
        total_reward = 0
        steps = 0
        
        while state != GOAL and steps < 100:  # 限制最大步数防止无限循环
            # 选择动作
            action = choose_action(state, q_table)
            # 执行动作
            next_state, reward = step(state, ACTIONS[action])
            total_reward += reward
            
            # Q-learning更新规则
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1], :])
            new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
            q_table[state[0], state[1], action] = new_value
            
            state = next_state
            steps += 1
        
        rewards.append(total_reward)
        
        # 每100回合打印一次进度
        if (episode + 1) % 100 == 0:
            print(f"回合 {episode + 1}/{EPISODES}, 总奖励: {total_reward:.2f}, 步数: {steps}")
    
    return q_table, rewards

if __name__ == "__main__":
    # 运行Q-learning算法
    q_table, rewards = run_q_learning()
    
    # 打印学习到的策略
    print_policy(q_table)
    
    # 可视化结果
    plot_results(rewards, q_table)
    
    # 演示最优路径
    print("\n最优路径演示:")
    state = START
    path = [state]
    while state != GOAL:
        action = np.argmax(q_table[state[0], state[1], :])
        state, _ = step(state, ACTIONS[action])
        path.append(state)
    
    print(" -> ".join([f"({i},{j})" for i, j in path]))
    print(f"路径长度: {len(path) - 1}步")
