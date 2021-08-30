We need an environment wrapper to provide a gym-alike environment for science birds. It should be have:

1. step -> next state, reward, if_done, info: Step the environment by one timestep. Returns observation, reward, done,
   info. info includes [did_win and total score]
2. reset -> state: Reset the environment's state. Returns observation.
3. render: render(self, mode='human'): Render one frame of the environment. The default mode will do something human
   friendly, such as pop up a window.
4. close: close the environment.
5. make(level = 1): launch the environment with a set of args. connects the agent to the game and start level.

SBAgent: class that can be used by all SB agents

1. a level selection function that returns level
2. a select_action function that given state return select_action

RLAgent: class that can be used by RL SB agents, inherited from SBAgent

1. degree to release point for discrete action space
2. training_action_selection with random eps
3. testing action selection just the optimal value
4. memory
5. update
6. training mode
7. testing mode
8. memory and model should be accessible for multiple agents, while we can have multiple agents running on the env and
   get memory.

We can have a general RLAgent class with

1. collect memory: open multiple agent and collect state value pairs for training
2. update parameter: sample from memory to update agent's model
3. test agent: test the agent on number of levels We need to initialise 1.memory, 2.agent with select_action, 3.env

For the above to work, we need an AgentThread to be inherited from threading.Thread. In the thread, agents need to be
assigned the levels that are required to run, the memory object. The AgentThread can be initialised by the agent used in
RLAgent. Running configs need to be given to the Thread.

20 may:

1. implement training/testing: agents needs to be able to choose the folder to train and the folder to test. -done
2. implement the training testing framework for dqn. -done
3. implement PRE. Only trainning should be changed. -done
4. implement Noisy Network for explorations -action selection would do the same

21 may:
previous:

1. done.
2. done to do:
3. implement PRE. Only training should be changed.

22 may

1. updated training test frame working. now with every 10 steps the agent will be tested and the model will be saved.
2. updated in config the type of reward, moved training to DQNBase to be more intuitive.

27 May todo:

1. update the gt to image function in gt reader, and update resnet in symbolic one - done
2. update agent to select action from a distribution - done
3. analysis the output to see if there's a problem of imbalance dataset - done
4. SOLVE THE DATA IMAGBLANCE PROBLEM IF THERE'S ONE - done

31 May Started final evaluation. todo:

1. add saved model folder to git and tell them to send also the model in the folder to me. - done
2. write a script to load saved models and test the agent for offline performance. - done
3. can use softmax to select action in testing time - done
4. add a random agent - done
5. tidy the code - half done
6. add return to step when level stuck - done

2 June

1. implemented offline testing for cross template testing, need to run test on it - done
2. write a script to automatically analysis all results we have and present them done by vimu

4 June

1. add network name and agent name to tensorboard file for future testing - done
2. add distributional action selection to agent - done

7 June 

at we need in the future is 
1) more agents to run. For 1), we can have PPO, A2C, continuous action space.
2) different within capability testing setting.  For 2), we can have train on 1 template test on the rest, train on half
template and test and train on all except 1 and test on the rest.

3). add random seed to ensure reproducible results add cuda to make sure use 1 if available. - not needed
4). change parameter of test attempts - not needed anymore

For the paper:
it the discussion, we need to show what the results mean, what does it take for the agent to achieve good performance,
why does it work, how it works
difference between good at playing angry birds vs good at physical reasoning
we are testing physical reasoning capability, may use agent in phyre, agent in angry birds... do a comparision


