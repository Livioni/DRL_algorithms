from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.Gym import build_env
import gym
gym.logger.set_level(40) # Block warning
import matplotlib as plt
import cv2

agent = AgentPPO()
env = build_env('Pendulum-v0')
args = Arguments(env, agent)

args.gamma = 0.97
args.net_dim = 2 ** 8
args.worker_num = 2
args.reward_scale = 2 ** -2
args.target_step = 200 * 16  # max_step = 200

args.eval_gap = 2 ** 5
train_and_evaluate(args)


img = cv2.imread(f"/content/{args.cwd}/plot_learning_curve.jpg", cv2.IMREAD_UNCHANGED)
plt.show(img)