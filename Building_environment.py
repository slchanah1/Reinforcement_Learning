import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

style.use("ggplot")  # setting our style!

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 300  # feel free to tinker with these!
FOOD_REWARD = 25  # feel free to tinker with these!
epsilon = 0.5  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through env visually.

# if we have a pickled Q table, we'll put the filename of it here.
start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def move(self, x=None, y=None):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x > SIZE-1:
            self.x = SIZE-1
        elif self.x < 0:
            self.x = 0

        if self.y > SIZE-1:
            self.y = SIZE-1
        elif self.y < 0:
            self.y = 0

    def action(self, choice):
        if choice == 0:
            self.move(1, 1)
        elif choice == 1:
            self.move(-1, -1)
        elif choice == 2:
            self.move(-1, 1)
        elif choice == 3:
            self.move(1, -1)


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    q_table[((i, ii), (iii, iiii))
                            ] = np.random.uniform(-5, 0, 4)
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    l = [player, food, enemy]

    for i in range(0, len(l)-1):
        for j in range(i+1, len(l)):
            while l[i].x == l[j].x and l[i].y == l[j].y:
                l[j].move()

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(
            f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        # print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy)  # new observation
        max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
        current_q = q_table[obs][action]  # current Q for our chosen action

        if reward == FOOD_REWARD:
            q_table[obs][action] = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            q_table[obs][action] = -ENEMY_PENALTY
        else:
            q_table[obs][action] = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        if show:
            # starts an rbg of our size
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            # sets the food location tile to green color
            env[food.x][food.y] = d[FOOD_N]
            # sets the player tile to blue
            env[player.x][player.y] = d[PLAYER_N]
            # sets the enemy location to red
            env[enemy.x][enemy.y] = d[ENEMY_N]
            # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = Image.fromarray(env, 'RGB')
            # resizing so we can see our agent in all its glory.
            img = img.resize((300, 300))
            cv2.imshow("image", np.array(img))  # show it!
            # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward

        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones(
    (SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
