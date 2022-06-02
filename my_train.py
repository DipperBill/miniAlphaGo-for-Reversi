# 收集数据， augment
from __future__ import print_function

import random
#
# from game import Game
#
# import mcts_alphaZero
# from mcts_pure import MCTS
# import mcts_pure
# # from mcts_pure import policy_value_fn
# import pickle
from game import Game
import numpy as np
from board import Board
from my_mcts_net import MCTSPlayer
from collections import defaultdict, deque

from AIplayer import  AIPlayer
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from my_net import PolicyValueNet
from copy import deepcopy


class TrainPipe:
    def __init__(self, init_model=None):
        # self.p1 = AIPlayer(color="X", n_playout=20000,type=1)
        # self.p2 = AIPlayer(color="O", n_playout=20000,type=0)
        # self.game = Game(self.p1, self.p2)
        self.board_width = 8
        self.board_height = 8

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 2000  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player_1 = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout,
                                        is_selfplay=1,
                                        color="X")
        self.mcts_player_2 = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout,
                                        is_selfplay=1,
                                        color="O")
        # self.mcts_player_2 = white_player = AIPlayer(color="O", n_playout=20000,
        #                                              type=0)

        self.game = Game(self.mcts_player_1, self.mcts_player_2)

        self.current_player = self.mcts_player_1

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    @staticmethod
    def get_current_state(board, color):
        square_state = np.zeros((3, 8, 8))

        mat_board = np.array(board._board)
        square_state[0][mat_board == "X"] = 1.
        square_state[1][mat_board == "O"] = 1.
        # square_state[:][mat_board=="X"] = 1
        if color == "X":
            square_state[2][:, :] = 1.

        return square_state[:, ::-1, :]

    @staticmethod
    def num_board(action):
        """
        数字坐标转化为棋盘坐标
        :param action:数字坐标 ,比如(0,0)
        :return:棋盘坐标，比如 （0,0）---> A1
        """
        row, col = action
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        if col in l and row in l:
            return chr(ord('A') + col) + str(row + 1)

    def switch_player(self, black_player, white_player):
        """
        游戏过程中切换玩家
        :param black_player: 黑棋
        :param white_player: 白棋
        :return: 当前玩家
        """
        # 如果当前玩家是 None 或者 白棋一方 white_player，则返回 黑棋一方 black_player;
        if self.current_player is None:
            return black_player
        else:
            # 如果当前玩家是黑棋一方 black_player 则返回 白棋一方 white_player
            if self.current_player == black_player:
                return white_player
            else:
                return black_player

    def collect_selfplay_data(self):
        """collect self-play data for training"""

        self.board = Board()
        self.current_player.mcts.color = "X"
        self.current_player.mcts.update_with_move(-1)
        print("begin a new game")
        print()
        print()

        state, mcts_probs, current_player = [], [], []
        # color = "X"
        map_color = {"X": "O", "O": "X"}
        while True:

            legal_action = list(self.board.get_legal_actions(self.current_player.mcts.color))
            if len(legal_action) == 0:
                # print("length = 0")
                if not len(list(self.board.get_legal_actions('X'))) and not len(list(self.board.get_legal_actions('O'))):
                    # print("break ???!")
                    winner, diff = self.board.get_winner()
                    self.current_player.mcts.update_with_move(-1)
                    break
                else:
                    # print("color: ", color)
                    # print("X:",len(list(self.board.get_legal_actions('X'))))
                    # print("O:",len(list(self.board.get_legal_actions('O'))))
                    # color = map_color[color]
                    self.current_player.mcts.color = map_color[self.current_player.mcts.color]
                    continue
            board = deepcopy(self.board._board)
            # 这里的返回值 变成了 数字； 需要转换成字符或者二维坐标
            move, move_probs = self.current_player.get_move(self.board, return_prob=1)
            self.board.display()
            flag = self.board._move(move, self.current_player.mcts.color)

            state.append(self.get_current_state(self.board, self.current_player.mcts.color))
            mcts_probs.append(move_probs)
            current_player.append(self.current_player.mcts.color)

            # print(move,"move_0")
            #
            # # 转换
            # move = move // 8, move % 8
            # print(move, "move1")
            # move = self.num_board(move)
            # print(move,"move_2")
            if flag:
                # color = map_color[color]
            # 一个玩家来对弈，
                self.current_player.mcts.color = map_color[self.current_player.mcts.color]
            # self.current_player = self.switch_player(self.mcts_player_1, self.mcts_player_2)
            # 判断游戏结束
            # end.winner = self.game.game_over()
        out_z = np.zeros(len(state))
        out_z[np.array([current_player]) == winner] = 1.
        print("here collect self play data done")
        # winner, diff = self.board.get_winner()
        return zip(state, mcts_probs, out_z)

    def read_data2Buffer(self,n_games=1):
        for i in range(n_games):
            play_data = self.collect_selfplay_data()
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_value = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate*self.lr_multiplier
            )
            new_probs, new_value = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_value.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_value.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         color="X")
        pure_mcts_player = AIPlayer(color="O", n_playout=self.n_playout, type=0)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            game = Game(current_mcts_player, pure_mcts_player)
            game.run()
            winner, diff = game.board.get_winner()
            win_cnt[winner] += 1
        test = 0 # 测试的是黑色方 黑色 = 0， 白色 = 1 平局 = 2
        win_ratio = 1.0*(win_cnt[test] + 0.5*win_cnt[2]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[0], win_cnt[1], win_cnt[2]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                print("begin to read data into buffer")
                self.read_data2Buffer(self.play_batch_size)
                print("end to read data into buffer")
                # self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    print("here we goto update")
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipe()
    training_pipeline.run()
