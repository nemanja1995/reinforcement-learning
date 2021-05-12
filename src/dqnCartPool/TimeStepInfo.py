class TimeStepInfo:
    def __init__(self, state, reward, new_state, done, action, step_num):
        self.action = action
        self.step_num = step_num
        self.done = done
        self.new_state = new_state
        self.reward = reward
        self.state = state
