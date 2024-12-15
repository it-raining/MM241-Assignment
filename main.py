import gymnasium as gym
import pygame
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
from student_submissions.cutting_stock import CuttingStockEnv

# pygame.init()
# pygame.display.set_mode((1000, 1000))
# Create the environment
env = CuttingStockEnv(
    "human",
    50, 50,
    100, 100,
    50,
    25,
    100,
    42
)
NUM_EPISODES = 100

if __name__ == "__main__":
    observation, info = env.reset(seed=42)
    # print(observation)

    policy2210xxx = Policy2210xxx(policy_id=1)
    for _ in range(200):
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(info)

        if terminated or truncated:
            print(info)
            observation, info = env.reset()
        
        pygame.display.update()
    observation, info = env.reset(seed=42)
    print(info)
env.close()
