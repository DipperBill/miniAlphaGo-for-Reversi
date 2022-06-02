import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # print("fuck u")
        # print(list(action_priors))
        for action, prob in list(action_priors):
            # print("expand from zip")
            # print(action, prob)
            if action not in self._children:
                # print("子节点 增加了没")
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000, color="X"):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.color = color

    def _playout(self, state):
        """
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """

        node = self._root
        color = self.color
        map_color = {"X": "O", "O": "X"}
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state._move(action, color)
            color = map_color[color]

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # 在这里没法传入颜色参数
        action_probs, leaf_value = self._policy(state, color)
        # print("action_probs, leaf_value")
        list_debug = list(action_probs)
        # print(action_probs,leaf_value)
        # Check for end of game.
        end, winner = self._game_end(state)
        if not end:
            # print("expand in mcts_net")
            # print(*action_probs)
            node.expand(list_debug)
            # self.color = map_color[self.color]
        else:
            # for end state，return the "true" leaf_value
            if winner == 2:  # tie
                leaf_value = 0.0
            else:
                if color == 'X':
                    player = 0
                else:
                    player = 1
                count_current = state.count(color)
                oppo_color = map_color[color]
                count_oppo = state.count(oppo_color)
                diff = np.abs(count_current - count_oppo)
                # diff = 1
                flag = int(winner == player) * 2 - 1
                leaf_value = flag * diff

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    @staticmethod
    def _game_end(state):
        """Check whether the game is ended or not
        """
        b_list = list(state.get_legal_actions('X'))
        w_list = list(state.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        winner, _ = state.get_winner()

        return is_over, winner

    def get_move_probs(self, state, temp=1e-3):
        """
        Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        # print("act_visits in my_mcts_net")
        # print(act_visits)

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0, color="X"):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, color=color)
        self._is_selfplay = is_selfplay
        self.color = color

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    @staticmethod
    def board_num(action):
        """
        棋盘坐标转化为数字坐标
        :param action:棋盘坐标，比如A1
        :return:数字坐标，比如 A1 --->(0,0)
        """
        row, col = str(action[1]).upper(), str(action[0]).upper()
        if row in '12345678' and col in 'ABCDEFGH':
            # 坐标正确
            x, y = '12345678'.index(row), 'ABCDEFGH'.index(col)
            return x, y

    # 这个地方充满随机性， 可以修改一下逻辑
    def get_move(self, board,  temp=1e-3, return_prob=0):

        map_color = {"X": "O", "O": "X"}
        if self._is_selfplay:
            self.color = map_color[self.color]
        sensible_moves = board.get_legal_actions(self.color)
        alist = list(sensible_moves)
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(64)
        if len(alist) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            acts_num = [self.board_num(item) for item in acts]
            # acts = [item[0] * 8 + item[1] for item in acts_num]
            # print(acts)
            # # 从字符型到 数字坐标
            # print("acts")
            acts_list = [item[0] * 8 + item[1] for item in acts_num]
            move_probs[list(acts_list)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
                # map_color = {"X": "O", "O": "X"}
                # self.mcts.color = map_color[self.mcts.color]
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                # print("move we choose", move)
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
