import gym
import numpy as np
import tqdm

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = EPISODES // 10
MAX_INT = 1000000000

epsilon = 1

EPISODE_START_DECAYING = 1
EPISODE_END_DECAYING = EPISODES//2
EPSILON_DECAY_RATE = epsilon/(EPISODE_END_DECAYING-EPISODE_START_DECAYING)


env = gym.make("MountainCar-v0")
env.reset()


print(" Environment information: ".center(150, "#"))
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

action_space_size = env.action_space.n
space_high = env.observation_space.high
space_low = env.observation_space.low

DISCRETE_OS_SIZE = [20] * len(space_high)
discrete_os_win_size = (space_high - space_low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [action_space_size]))

successful_episodes = 0
num_steps = 0

min_steps = MAX_INT
global_min = min_steps


def get_discrete_state(state):
    discrete_state = (state - space_low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# print(q_table[discrete_state])

render = False
for episode in tqdm.tqdm(range(EPISODES)):
    episode += 1
    if episode % SHOW_EVERY == 0:
        if min_steps == MAX_INT:
            min_steps = -1
        if global_min == MAX_INT:
            global_min = -1
        print("Episode: {:5} [rate: {:4}/{:4}] [min: {:4}\t glob min: {:4}]".format(episode, successful_episodes,
                                                                                    SHOW_EVERY, min_steps, global_min))
        if global_min == -1:
            global_min = MAX_INT
        successful_episodes = 0
        render = True
    else:
        render = False

    done = False
    discrete_state = get_discrete_state(env.reset())
    num_steps = 0
    min_steps = MAX_INT
    while not done:
        num_steps += 1
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=action_space_size)
        else:
            action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action=action)
        new_discrete_state = get_discrete_state(state=new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            successful_episodes += 1
            if num_steps < min_steps:
                min_steps = num_steps

            if num_steps < global_min:
                global_min = num_steps

        if EPISODE_START_DECAYING < episode < EPISODE_END_DECAYING:
            if epsilon > 0:
                epsilon -= EPSILON_DECAY_RATE

        discrete_state = new_discrete_state

print("END".center(150, " "))
env.close()


