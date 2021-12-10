import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]#计算新的位置
        new_position = self._limit_coordinates(new_position).astype(int)#限制动作不超出边界
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)#10*7的网格

        nS = np.prod(self.shape)#nS=70 表示70个状态数
        nA = 4 #动作数

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1#第3，4，5，8列风力为1
        winds[:,[6,7]] = 2#第6，7列风力为2

        # Calculate transition probabilities
        P = {}#P是一个字典{状态：{动作：[概率，下一状态，回报，是否结束]}}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)#将70个状态展平为一维列表，从二维矩阵中编号从左到右从上往下
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)#向上走，行减1
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)#向右走，列加1
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)#向下走，行加1
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)#向左走，列减1

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0 #从编号反求二维矩阵的（行，列） 所以索引都是从0开始

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")