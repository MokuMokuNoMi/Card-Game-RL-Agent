import unittest
import optimization
from gym import spaces
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_card_functions(self):
        self.assertEqual(optimization.suit(0), "♦️")
        self.assertEqual(optimization.to_full_card(0)[0], "A♦️")
        self.assertEqual(optimization.is_same_card(0, 3), True)
        self.assertEqual(optimization.is_same_card(0, 4), False)
        self.assertEqual(optimization.is_same_card(47, 51), False)
        self.assertEqual(optimization.is_same_card(47, 48), False)
        self.assertEqual(optimization.is_same_card(47, 46), True)
        self.assertEqual((optimization.is_same_card(34, 35)), True)
        self.assertEqual(optimization.is_same_card(37, 35), False)

        # pairs/triples/quads
        self.assertEqual(optimization.group_eval([[0, 1, 2, 3, 5]]), 2)
        self.assertEqual(optimization.group_eval([[0, 0, 0, 0, 1]]), 0)
        self.assertEqual(optimization.group_eval([[0, 3, 2, 5, 6]]), 0)
        self.assertEqual(optimization.group_eval([[0, 5, 6, 8, 9]]), 15)

        # straight
        self.assertEqual(optimization.group_eval([[0, 4, 8, 12, 16]]), 0)
        self.assertEqual(optimization.group_eval([[1, 4, 8, 12, 16]]), 0)
        self.assertEqual(optimization.group_eval([[2, 4, 8, 12, 16]]), 0)
        self.assertEqual(optimization.group_eval([[3, 4, 8, 12, 16]]), 0)
        self.assertEqual(optimization.group_eval([[51, 47, 43, 39, 35]]), 0)

        # 0-3 A, 4-7 2, 8-11 3, 12-15 4, 16-19 5, 20-23 6, 24-27 7, 28-31 8
        # 32-35 9, 36-39 10, 40-43 J, 44-47 Q, 48-51 K
        self.assertEqual(optimization.group_eval([[51, 47, 43, 39, 34]]), 0)
        self.assertEqual(optimization.group_eval([[51, 47, 43, 39, 0]]), 0)
        self.assertEqual(optimization.group_eval([[17, 27, 13, 20, 30]]), 0)

        # flush
        self.assertEqual(optimization.group_eval([[0, 8, 12, 20, 32]]), 0)

        # no match
        self.assertEqual(optimization.group_eval([[1, 8, 12, 20, 32]]), 37)
        self.assertEqual(optimization.group_eval([[0, 8, 12, 20, 32], [1, 8, 12, 20, 32]]), 37)
        self.assertEqual(optimization.group_eval([[0, 5, 6, 8, 9], [0, 8, 12, 20, 32], [1, 8, 12, 20, 32]]), 52)
        self.assertEqual(optimization.group_eval([[5, 33, 22, 51, 10], [50, 42, 17, 38, 12], [30, 4, 15, 14, 39],
                                                  [49, 11, 19, 35, 29], [9, 7, 36, 32, 20], [5, 50, 30, 49, 9],
                                                  [33, 42, 4, 11, 7], [22, 17, 15, 19, 36], [51, 38, 14, 35, 32],
                                                  [10, 12, 39, 29, 20], [5, 42, 15, 35, 20], [10, 38, 15, 11, 9]]), 500-191)

        # generate cards
        print(optimization.generate_cards())
        self.assertEqual(len(optimization.generate_cards()), 52)
        cards = optimization.generate_cards()
        self.assertEqual(np.equal(cards[0], 1) or np.equal(cards[0], -1), True)


if __name__ == '__main__':
    unittest.main()
