## Getting started

- Before we deep-dive into Reinforcement Learning (RL), let's go through some videos that showcase how RL is solving some really complicated real world problems:
  - Waymo 360 Experience of Autonomous Driving System

    <div class="video-wrapper">
    <iframe width="256" height="256" src="https://www.youtube.com/embed/B8R148hFxPw" frameborder="0" allowfullscreen></iframe>
    </div>

  - DeepMind AI that plays 57 Atari games - [Two Minute Papers](https://www.youtube.com/c/K%C3%A1rolyZsolnai)

    <div class="video-wrapper">
    <iframe width="256" height="256" src="https://www.youtube.com/embed/dJ4rWhpAGFI" frameborder="0" allowfullscreen></iframe>
    </div>


  - QT-Opt - Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation 

    <div class="video-wrapper">
    <iframe width="256" height="256" src="https://www.youtube.com/embed/W4joe3zzglU" frameborder="0" allowfullscreen></iframe>
    </div>

- As obvious from the videos, RL is used across domains - it can control cars, play games and even interact with real world using robot. Isn't it interesting?! :robot: 

- Now let's answer the first question - **What is Reinforcement Learning?** Contrary to basic supervised and unsupervised problems, RL is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. [1]
- I know this must be too much to unpack, so let's dissect some important keywords in the next section. 

## Important terms

- **Agent:** it is the entity that takes some action which is influenced by current scenario and it's past experience. We try to create an algorithm or AI model that can act as an agent. 
- **Environment:** this is where agent takes actions. It is the complete world wherein the agent lives and traverse. The environment considers the agent's actions and return rewards *(positive or negative based on type of impact)* and changes the state of the agent. An environment could be deterministic *(taking an action from a state leads to a fixed state always)* or non-deteministic *(state transition wrt action selection is probabilistic)*.

    ``` mermaid
    graph LR
    A[Agent] -- state 's' --> B[Environment];
    A -- action 'a' --> B;
    B -- reward 'r' --> A;
    B -- new state 'sn' --> A;
    ```

- **States:** An environment consists of multiple states, infact this denotes the different scenarios an agent could be in. States could be (1) engineered *(ex: In [MountainCar](https://github.com/openai/gym/wiki/MountainCar-v0) env, combination of calculated position and velocity forms the state)*, (2) raw *(ex: using screen images for video games)*, or (3) combination of both *(ex: autonomus driving system considering raw video feeds and some extracted features)*. States could be discrete *(ex: cell A to cell B)* or continous *(ex: screenshot of a game, no two images could be same)*.
- **Action:** These are the possible interactions an agents can make with the environment. Actions could differ wrt to the current state of the agent. It can also be discrete *(ex: left turn and right turn)* or continous *(ex: -120 degree turn)*.
- **Reward:** Agent performs some actions and enviroment returns rewards. This is usually programmed wrt the behavior we want the agent to learn. If we want the agent to reach a goal -- give +1 point at that point. If we want the agent to move faster -- give -1 for every step it takes *(so that it tries ot finish the game asap)*. Reward designing is a very important part of solving RL problem. 
- **Policy:** It is the rulebook that tells the agent what to do for a given state and possible set of actions. Policy based RL agents tries to directly learn the best policy for a given state, whereas value based RL agents learns the value function *(estimated reward for each action in a state)* and then select action based on their strategy. 
- **Episodes:** One complete iteration over the environment by the agent, either till it passed or failed, is called an epsiode. In game terminology, think of a round as one episode. 
<!-- - **Exploration vs Exploitation:** RL agent's policy fundamental waether to perform exploraton -->
<!-- - **Q-learning:** Q-learning is a classic RL algorithm where agent learns to maximize the value function. -->

!!! Hint
    If you are looking for a detailed course on Reinforcement learning, I cannot think of a better free course than the one by [David Silver on Youtube](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-). :ok_hand:


## References

[1] [Reinforcement Learning - Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)

