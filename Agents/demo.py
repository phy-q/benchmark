from SBAgent import SBAgent
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper

# for using reward as score and 50 times faster game play
env = SBEnvironmentWrapper(reward_type="score", speed=50)
level_list = [1, 2, 3]  # level list for the agent to play
dummy_agent = SBAgent(env=env, level_list=level_list)  # initialise agent
dummy_agent.state_representation_type = 'image'  # use symbolic representation as state and headless mode
env.make(agent=dummy_agent, start_level=dummy_agent.level_list[0],
         state_representation_type=dummy_agent.state_representation_type)  # initialise the environment

s, r, is_done, info = env.reset()  # get ready for running
for level_idx in level_list:
    is_done = False
    while not is_done:
        s, r, is_done, info = env.step([-100, -100])  # agent always shoots at -100,100 as relative to the slingshot

    env.current_level = level_idx+1  # update the level list once finished the level
    if env.current_level > level_list[-1]: # end the game when all game levels in the level list are played
        break
    s, r, is_done, info = env.reload_current_level() #go to the next level
