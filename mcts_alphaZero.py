import numpy as np
import copy
from operator import itemgetter
import time


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def rollout_policy_fn(board, color):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    available_action = list(board.get_legal_actions(color))
    # print(available_action)
    action_probs = np.random.rand(len(available_action))
    # action_probs = np.random.rand(len(board.availables))
    return zip(available_action, action_probs)


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
        for action, prob in action_priors:
            if action not in self._children:
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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, color="X"):
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
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        ## 这个地方可以尝试修改； 应该从当前state进行搜索
        node = self._root
        color = self.color
        map_color = {"X": "O", "O": "X"}
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            # 引入交换手 的规则
            action, node = node.select(self._c_puct)
            state._move(action, color)

            color = map_color[color]
            # else:
            #     print("no position: " + color)
            # state.display()
            # print("*"*15)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state, color)

        # Check for end of game.
        # 这个地方有问题
        end, winner = self._game_end(state)
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state, color)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, _color, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        if _color == 'X':
            player = 0
        else:
            player = 1
        # color = 'X' if player == 0 else 'O'
        color = _color
        # print(color)
        map_color = {"X": "O", "O": "X"}
        for i in range(limit):
            # print(color)
            end, winner = self._game_end(state)
            if end:
                break
            action_probs = list(rollout_policy_fn(state, color))
            # print(i,action_probs)
            if len(action_probs) == 0:
                break
            max_action = max(action_probs, key=itemgetter(1))[0]
            # print(max_action)
            state._move(max_action, color)

            color = map_color[color]

        # print(i)
        # print(i)
        if winner == 2:  # tie
            return 0
        else:
            count_current = state.count(color)
            oppo_color = map_color[color]
            count_oppo = state.count(oppo_color)
            diff = np.abs(count_current - count_oppo)
            # diff = 1
            flag = int(winner == player)*2-1
            leaf_value = flag * diff
            # print("evaluate")
            # print(color, self.color, leaf_value)
            return leaf_value

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
        """Run all playouts sequentially and return the available actions and
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

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        time_limit = 5
        T1 = time.time()

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            T2 = time.time()
            if T2 - T1 > time_limit:
                break
        # print(self._root._children.items())
        # print("here")
        # print(self._root._children.items())

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def __str__(self):
        return "MCTS"

