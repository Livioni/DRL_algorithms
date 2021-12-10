import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from blackjack import BlackjackEnv
import blackjack
env = BlackjackEnv()
def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise 玩家策略手牌数小于20继续叫牌
    return 0 if score >= 18 else 1

for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        print_observation(observation)
        action = strategy(observation)
        print("Taking action: {}".format( ["Stick", "Hit"][action]))
        observation, reward, done, _ = env.step(action)
        if done:
            print_observation(observation) 
            print('Dealer''s Score：',blackjack.sum_hand(env.dealer))
            print("Game end. Reward: {}\n".format(float(reward)))
            break
            