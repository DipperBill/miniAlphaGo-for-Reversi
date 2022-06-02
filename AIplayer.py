from __future__ import print_function
from board import Board
from mcts_alphaZero import MCTS
import mcts_alphaZero
from mcts_pure import MCTS
import mcts_pure
# from mcts_pure import policy_value_fn
import pickle
from game import Game
import numpy as np


def policy_value_fn(board, color):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    # color = "X"
    available_action = list(board.get_legal_actions(color))
    # print("available")
    # print(available_action)
    # print("color")
    # print(color)
    action_probs = np.ones(len(available_action)) / len(available_action)
    return zip(available_action, action_probs), 0


class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color, c_puct=5, n_playout=2000, type=1):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        :param c_puct: 超参数C
        :param n_playout: 总行动次数
        """

        self.color = color
        # print("init color:" , color)
        if type == 1:
            self.mcts = mcts_pure.MCTS(policy_value_fn, self.color, c_puct, n_playout)  # 初始化MCTS
        else:
            self.mcts = mcts_alphaZero.MCTS(policy_value_fn, c_puct, n_playout, self.color)  # 初始化MCTS

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        # 合法行动
        action_list = list(board.get_legal_actions(self.color))
        if len(action_list) == 0:
            return None  # 无子可落
        action = self.mcts.get_move(board)
        self.mcts.update_with_move(-1)
        # ------------------------------------------------------------------------

        return action


# 导入随机包
import random


class RandomPlayer:
    """
    随机玩家, 随机返回一个合法落子位置
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def random_choice(self, board):
        """
        从合法落子位置中随机选一个落子位置
        :param board: 棋盘
        :return: 随机合法落子位置, e.g. 'A1' 
        """
        # 用 list() 方法获取所有合法落子位置坐标列表
        action_list = list(board.get_legal_actions(self.color))

        # 如果 action_list 为空，则返回 None,否则从中选取一个随机元素，即合法的落子坐标
        if len(action_list) == 0:
            return None
        else:
            return random.choice(action_list)

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        action = self.random_choice(board)
        return action


class HumanPlayer:
    """
    人类玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘输入人类合法落子位置
        :param board: 棋盘
        :return: 人类下棋落子位置
        """
        # 如果 self.color 是黑棋 "X",则 player 是 "黑棋"，否则是 "白棋"
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人类玩家输入落子位置，如果输入 'Q', 则返回 'Q'并结束比赛。
        # 如果人类玩家输入棋盘位置，e.g. 'A1'，
        # 首先判断输入是否正确，然后再判断是否符合黑白棋规则的落子位置
        while True:
            action = input(
                "请'{}-{}'方输入一个合法的坐标(e.g. 'D3'，若不想进行，请务必输入'Q'结束游戏。): ".format(player,
                                                                             self.color))

            # 如果人类玩家输入 Q 则表示想结束比赛
            if action == "Q" or action == 'q':
                return "Q"
            else:
                row, col = action[1].upper(), action[0].upper()

                # 检查人类输入是否正确
                if row in '12345678' and col in 'ABCDEFGH':
                    # 检查人类输入是否为符合规则的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的输入不合法，请重新输入!")


# 人类玩家黑棋初始化
#
# black_player = AIPlayer(color="X",n_playout=20000,type=1)
# # # black_player = AIPlayer(color="X",n_playout=20000,type=0)
# # # black_player = RandomPlayer(color="X")
# #
# # # AI 玩家 白棋初始化
# # # white_player = AIPlayer(color="O", n_playout=20000,type=0)
# white_player = AIPlayer(color="O", n_playout=20000,type=0)
# #
# # # 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
# game = Game(black_player, white_player)
# #
# # # 开始下棋
# game.run()
