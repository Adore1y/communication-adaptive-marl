#!/usr/bin/env python3
"""
多智能体机器人环境
模拟仓库搬运任务
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class MultiAgentWarehouseEnv(gym.Env):
    """
    多智能体仓库环境
    多个机器人协作搬运物品
    """
    
    def __init__(self, num_agents=10):
        super().__init__()
        
        self.num_agents = num_agents
        self.physics_client = None
        
        # 观测空间: [位置(2), 速度(2), 朝向(1), 物品状态(1)]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # 动作空间: [前进(1), 转向(1)]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        """重置环境"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # 连接物理引擎（可视化模式）
        self.physics_client = p.connect(p.DIRECT)  # 或 p.GUI
        
        # 加载场景
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        
        # 初始化机器人
        self.robots = []
        for i in range(self.num_agents):
            robot_id = p.loadURDF("r2d2.urdf", 
                                   basePosition=[i*2, i*2, 0],
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
            self.robots.append(robot_id)
        
        # 随机生成任务
        self.targets = np.random.uniform(-10, 10, (self.num_agents, 2))
        self.carrying = [False] * self.num_agents
        
        # 初始观测
        observations = [self._get_observation(i) for i in range(self.num_agents)]
        
        return np.array(observations), {}
    
    def step(self, actions):
        """执行一步"""
        # 执行动作
        for i, (robot_id, action) in enumerate(zip(self.robots, actions)):
            if self.physics_client is not None:
                # 简化的运动控制
                forward = action[0] * 0.1
                turn = action[1] * 0.1
                
                # 更新机器人位置
                pos, orn = p.getBasePositionAndOrientation(robot_id)
                # (简化版，实际需要更复杂的动力学)
        
        # 物理步进
        p.stepSimulation()
        
        # 计算奖励
        rewards = []
        dones = []
        
        for i in range(self.num_agents):
            reward = self._compute_reward(i)
            done = self._check_done(i)
            rewards.append(reward)
            dones.append(done)
        
        # 新观测
        observations = [self._get_observation(i) for i in range(self.num_agents)]
        
        return np.array(observations), np.array(rewards), np.all(dones), {}
    
    def _get_observation(self, agent_id):
        """获取智能体观测"""
        if self.physics_client is None:
            return np.zeros(self.observation_space.shape)
        
        robot_id = self.robots[agent_id]
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        vel, ang_vel = p.getBaseVelocity(robot_id)
        
        # 简化的观测
        obs = np.array([
            pos[0], pos[1],              # 位置
            vel[0], vel[1],              # 速度
            orn[2],                      # 朝向
            1.0 if self.carrying[agent_id] else 0.0  # 是否携带物品
        ])
        
        return obs
    
    def _compute_reward(self, agent_id):
        """计算奖励"""
        robot_id = self.robots[agent_id]
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        # 距离目标的奖励
        target = self.targets[agent_id]
        dist = np.sqrt((pos[0]-target[0])**2 + (pos[1]-target[1])**2)
        reward = -dist * 0.1
        
        # 到达目标额外奖励
        if dist < 0.5:
            reward += 10.0
        
        return reward
    
    def _check_done(self, agent_id):
        """检查是否完成"""
        # 简化版：固定时间步
        return False
    
    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


