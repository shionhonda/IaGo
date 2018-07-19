import unittest
import MCTS

class TestStringMethods(unittest.TestCase):

    def test(self):
        self.tree = MCTS.GameTree(MCTS.Node("root"))
        self.tree.make_child("1")
        #self.tree.make_child("2")
        self.tree.move_down()
        self.tree.make_child("3")
        self.tree.move_up()
        self.assertTrue(self.tree.cur.is_root())
        self.assertFalse(self.tree.cur.is_leaf())
        self.tree.move_down()
        self.assertFalse(self.tree.cur.is_root())
        self.assertFalse(self.tree.cur.is_leaf())
        self.tree.move_down()
        self.assertFalse(self.tree.cur.is_root())
        self.assertTrue(self.tree.cur.is_leaf())
