Env = C4Env()

episodes = 10
for episode in range(1, episodes+1):
    state = Env.reset()
    done = False
    score = 0

    while not done:
        Env.render()
        action = Env.action_space.sample()
        n_state, reward, done = Env.step(action)
        score+=reward
    print("Episode:{},Score:{}".format(episode, score))