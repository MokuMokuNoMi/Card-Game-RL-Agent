from copy import deepcopy
import gym
import numpy as np
import math
from gym import spaces
import random


def is_same_card(card_a, card_b) -> bool:
    if abs(card_a - card_b) <= 3:
        if card_a > card_b:
            if card_a % 4 > card_b % 4:
                return True
            return False
        else:
            if card_b % 4 > card_a % 4:
                return True
            return False
    return False


def generate_cards():
    shuffled = np.empty(22)
    shuffled.fill(.1)
    shuffled = np.append(shuffled, np.ones(30))
    random.shuffle(shuffled)
    return shuffled


def suit(card) -> int:
    if card % 4 == 0:
        return "♦️"
    if card % 4 == 1:
        return "♠"
    if card % 4 == 2:
        return "♥"
    return "♣️"


def to_card(card):
    if card < 4:
        return "A", 1
    if card < 8:
        return "2", 2
    if card < 12:
        return "3", 3
    if card < 16:
        return "4", 4
    if card < 20:
        return "5", 5
    if card < 24:
        return "6", 6
    if card < 28:
        return "7", 7
    if card < 32:
        return "8", 8
    if card < 36:
        return "9", 9
    if card < 40:
        return "10", 10
    if card < 44:
        return "J", 11
    if card < 48:
        return "Q", 12
    return "K", 13


def to_full_card(card):
    if card < 4:
        return "A" + suit(card), 15
    if card < 8:
        return "2" + suit(card), 2
    if card < 12:
        return "3" + suit(card), 3
    if card < 16:
        return "4" + suit(card), 4
    if card < 20:
        return "5" + suit(card), 5
    if card < 24:
        return "6" + suit(card), 6
    if card < 28:
        return "7" + suit(card), 7
    if card < 32:
        return "8" + suit(card), 8
    if card < 36:
        return "9" + suit(card), 9
    if card < 40:
        return "10" + suit(card), 10
    if card < 44:
        return "J" + suit(card), 10
    if card < 48:
        return "Q" + suit(card), 10
    return "K" + suit(card), 10


def count_same_card(group, card) -> int:
    sig = to_card(card)[0]
    amt = 0
    for i in range(len(group)):
        if to_card(group[i])[0] == sig:
            amt += 1
    return amt


def is_royal_straight(group):
    if group[0] != 1:
        return False
    if group[1] != 10:
        return False
    if group[2] != 11:
        return False
    if group[3] != 12:
        return False
    if group[4] != 13:
        return False
    return True


def group_eval(groups) -> int:
    rew = 0
    for i in groups:

        # Reconstruct list without duplicates
        good_lst = []
        for j in range(len(i)):
            count = count_same_card(i, i[j])
            if count == 1:
                good_lst.append(i[j])
        # If len < 5, duplicates exist, so points can't be avoided in other categories
        if len(good_lst) < 5:
            for element in good_lst:
                rew += to_full_card(element)[1]
        else:
            s = suit(i[0])
            indicator = 1
            for j in range(1, 5):
                if suit(i[j]) != s:
                    indicator = 0
            if indicator == 1:
                # 5 card flush, add 0 points
                rew += 0
            else:
                i.sort()
                # Detect if a straight exists
                is_straight = True
                card_num_lst = []
                for num in i:
                    card_num_lst.append(to_card(num)[1])
                card = card_num_lst[0]
                for j in range(1, len(i)):
                    if card_num_lst[j] != card + 1:
                        is_straight = False
                    card = card_num_lst[j]
                if is_straight or is_royal_straight(card_num_lst):
                    rew += 0
                else:
                    for element in i:
                        rew += to_full_card(element)[1]
    return rew


class OptimEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # can modify init to take in a config
    def __init__(self):
        # state has 52 binary values representing the card (0 if used) and 25 representing the grid
        # state[0-51] represents cards, state[52-76] represents grid
        self.cards = generate_cards()
        self.board = np.full((5, 5), .1)
        self.reward = 0
        self.time = 0
        self.points_eval = None
        self.chose_valid = False
        self.curr_card = -1
        self.steps_per_game = 50
        self.done = False
        self.cards_filled = 0
        # action space is a tuple with the first element representing cards to play, and the second a space on the
        # 5x5 grid of cards
        open_deck_spaces = self.deck_open_spaces()
        open_board_spaces = self.board_open_spaces()
        self.action_space = spaces.MultiDiscrete([max(len(open_deck_spaces), 1), max(len(open_board_spaces), 1)])
        # self.action_space = spaces.MultiDiscrete([52, 25])
        self.observation_space = spaces.Dict(
            spaces={
                "cards": spaces.MultiBinary(52),
                "board": spaces.MultiDiscrete([52] * 25),
            }
        )
        self.info = {"successful_moves": 0, "invalid_board_selection": 0, "invalid_card_selection": 0,
                     "games_finished": 0, "points_from_finishing": 0}

    def deck_open_spaces(self):
        open_spaces = []
        for i in range(len(self.cards)):
            if np.equal(self.cards[i], 1):
                open_spaces.append(i)
        return open_spaces

    def board_open_spaces(self):
        open_board_spaces = []
        for i in range(5):
            for j in range(5):
                if np.equal(self.board[i][j], .1):
                    open_board_spaces.append(i*5 + j)
        return open_board_spaces

    def step(self, action: list):

        # set a max turn count

        self.reward = 0
        action[0] = max(0, min(action[0], len(self.deck_open_spaces())-1))
        action[1] = max(0, min(action[1], len(self.board_open_spaces())-1))
        card = self.deck_open_spaces()[action[0]]
        grid_space = self.board_open_spaces()[action[1]]
        card = max(0, min(card, 51))
        grid_space = max(0, min(grid_space, 24))
        # grid ranges from 52-76 in the state
        row, col = grid_space//5, grid_space % 5

        if self.time >= self.steps_per_game:
            self.done = True
            obs = {
                "cards": self.cards,
                "board": self.board.flatten()
            }
            return obs, self.reward, self.done, self.info

        invalid = False
        if not np.equal(self.board[row, col], .1):
            invalid = True
            self.info['invalid_board_selection'] += 1
            # space filled by previous card, invalid move
            self.reward -= 23
        if np.equal(self.cards[card], .1):
            invalid = True
            # card has already been placed on grid or wasn't in deck at all, invalid selection
            self.reward -= 18
            self.info['invalid_card_selection'] += 1
        if not invalid:
            self.info['successful_moves'] += 1

            # To debug board issue
            self.chose_valid = True
            self.curr_card = card

            self.reward += 50 * (self.cards_filled+25)/50
            # self.reward += 30
            self.board[row, col] = card
            self.cards[card] = .1
            self.cards_filled += 1
            if self.cards_filled == 25:
                rew = 500 - self.point_evaluation()
                print("NUM POINTS GAINED")
                print(rew)
                self.reward += rew
                self.info['points_from_finishing'] += rew
                self.info['games_finished'] += 1
                self.done = True
        obs = {
            "cards": self.cards,
            "board": self.board.flatten(),
        }
        open_deck_spaces = self.deck_open_spaces()
        open_board_spaces = self.board_open_spaces()
        self.action_space = spaces.MultiDiscrete([max(len(open_deck_spaces), 1), max(len(open_board_spaces), 1)])
        self.time += 1
        return obs, self.reward, self.done, self.info

    # outputs a reward
    def point_evaluation(self) -> int:
        # iterates through all diagonals/rows/columns
        # grid ranges from 30-54 in the state
        rows = []
        rew = 0
        for i in range(0, 5):
            rows.append([self.board[i][0], self.board[i][1], self.board[i][2], self.board[i][3],
                         self.board[i][4]])
        columns = []
        columns.append([self.board[0][0], self.board[1][0], self.board[2][0], self.board[3][0],
                        self.board[4][0]])
        columns.append([self.board[0][1], self.board[1][1], self.board[2][1], self.board[3][1],
                        self.board[4][1]])
        columns.append([self.board[0][2], self.board[1][2], self.board[2][2], self.board[3][2],
                        self.board[4][2]])
        columns.append([self.board[0][3], self.board[1][3], self.board[2][3], self.board[3][3],
                        self.board[4][3]])
        columns.append([self.board[0][4], self.board[1][4], self.board[2][4], self.board[3][4],
                        self.board[4][4]])
        diagonals = []
        diag1 = [self.board[0][0], self.board[1][1], self.board[2][2], self.board[3][3], self.board[4][4]]
        diag2 = [self.board[0][4], self.board[1][3], self.board[2][2], self.board[3][1], self.board[4][0]]
        diagonals.append(diag1)
        diagonals.append(diag2)
        rew += group_eval(columns)
        rew += group_eval(diagonals)
        rew += group_eval(rows)
        return rew

    def reset(self):
        if self.cards_filled == 25:
            print("BOARD FINISHED")
            print(self.board)
        self.cards = generate_cards()
        self.board = np.full((5, 5), .1)
        self.steps_per_game = 50
        self.done = False
        self.chose_valid = False
        self.curr_card = -1
        self.time = 0
        self.reward = 0
        self.cards_filled = 0
        self.points_eval = None
        open_deck_spaces = self.deck_open_spaces()
        open_board_spaces = self.board_open_spaces()
        self.action_space = spaces.MultiDiscrete([max(len(open_deck_spaces), 1), max(len(open_board_spaces), 1)])
        # self.action_space = spaces.MultiDiscrete([52, 25])
        self.info = {"successful_moves": 0, "invalid_board_selection": 0, "invalid_card_selection": 0,
                     "games_finished": 0, "points_from_finishing": 0}
        return {
            "cards": self.cards,
            "board": self.board.flatten(),
        }

    def render(self, mode="Human"):
        if self.chose_valid:
            print("Chose a valid card:" + str(self.curr_card))
        elif self.cards_filled == 25:
            print("Filled Board!")
        elif self.done is True:
            print("Reached turn limit.")
        else:
            print("Did not choose a valid card")
        self.chose_valid = False
        self.curr_card = -1
        print("Cards Remaining: ")
        print(np.where(self.cards == 1)[0])
        print("-----")
        print("Grid: ")
        print(self.board)
        print("================================")
        if self.points_eval:
            print(self.points_eval)
