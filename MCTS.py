import numpy as np
import copy
import chainer
from chainer import serializers, cuda, optimizers, Variable
import policy
import value
import mcts_self_play
from game import GameFunctions as gf

class Node(object):

    def __init__(self, parent=None, prob=0):
        self.parent = parent
        self.children = {}  # Dictionary of Node with key:action
        self.n_visits = 0
        self.Q = 0
        # This value for u will be overwritten in the first call to update()
        self.u = prob
        self.P = prob

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children)<1

    def expand(self, action_probs):
        """Expand tree by creating new children.
        Arguments:
        action_probs -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.
        Returns:
        None
        """
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        Returns:
        None
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += (leaf_value - self.Q) / self.n_visits
        # Update u, the prior weighted by an exploration hyperparameter c_puct
        # Note that u is not normalized to be a distribution.
        if not self.is_root():
            self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)

    def update_recursive(self, leaf_value, c_puct):
        # If it is not root, this node's parent should be updated first.
        if self.parent is not None:
            self.parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        return self.Q + self.u

class MCTS(object):

    def __init__(self, lmbda=0.5, c_puct=5, playout_depth=5, n_playout=243):
        self.root = Node(None, 1.0)
        self.policy_net = policy.SLPolicy()
        serializers.load_npz('./models/sl_model.npz', self.policy_net)
        self.value_net = value.ValueNet()
        serializers.load_npz('./models/value_model.npz', self.value_net)
        chainer.config.train = False
        chainer.config.enable_backprop = False
        self.lmbda = lmbda
        self.c_puct = c_puct
        self.L = playout_depth
        self.n_playout = n_playout

    def policy_func(self, state, color, actions):
        state_var = gf.make_state_var(state, color)
        prob = self.policy_net(state_var).data.reshape(64)
        action_probs = []
        for action in actions:
            action_probs.append((action, prob[action]))
        return action_probs

    def value_func(self, state, color):
        state_var = gf.make_state_var(state, color)
        return self.value_net(state_var).data.reshape(1)

    def playout(self, state, leaf_depth, color):
        node = self.root
        c = color
        for i in range(leaf_depth):
            if node.is_leaf():
                # a list of tuples of actions and their prior probability
                actions = gf.legal_actions(state, c)
                if len(actions)<1:
                    break
                action_probs = self.policy_func(state, c, actions)
                node.expand(action_probs)
            # Greedily select next move.
            action, node = node.select()
            state = gf.place_stone(state, action, c)
            c = 3-c
        v = self.value_func(state, color) if self.lmbda < 1 else 0
        z = self.evaluate_rollout(state, color) if self.lmbda > 0 else 0
        leaf_value = (1-self.lmbda)*v + self.lmbda*z

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value, self.c_puct)

    def evaluate_rollout(self, state, color):
        sim = mcts_self_play.Simulate(state)
        return sim(color)

    def get_move(self, state, color):
        for n in range(self.n_playout):
            state_copy = state.copy()
            self.playout(state_copy, self.L, color)
        # chosen action is the *most visited child*, not the highest-value
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
