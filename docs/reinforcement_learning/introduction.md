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

## The Paradigms in RL

### Forward Reinforcement Learning

- It refers to the process of learning a policy or control strategy in an environment by taking actions and receiving feedback or rewards from the environment. The objective of this approach is to maximize the expected cumulative reward obtained from the environment by learning a policy that maps the state of the environment to an action to be taken.

### Inverse Reinforcement Learning

- Inverse Reinforcement Learning (IRL) aims to learn the underlying reward function of a given environment from a set of observed trajectories, instead of learning a policy directly. This approach can be used to infer the reward function from human demonstrations or expert behavior, and then learn a policy that maximizes the inferred reward function.

### Behavior Cloning

- Behavior cloning refers to a supervised learning technique where a model is trained to mimic the behavior of an expert agent or human in a given task. In this approach, the model is trained on a dataset of state-action pairs, where each action is a direct imitation of the expert's actions in the corresponding state. Behavior cloning can be used as a pre-training step for more complex reinforcement learning algorithms, or as a standalone approach in cases where the expert's behavior is sufficient for solving the task.

### Apprenticeship Learning

- Apprenticeship learning, also known as imitation learning, is a machine learning technique where a learner tries to imitate the behavior of an expert agent or human in a given task. Unlike behavior cloning, which only learns to mimic the expert's actions, apprenticeship learning aims to learn the underlying decision-making process or policy of the expert by observing their behavior in a given environment. This approach is useful when the expert's behavior is not easily captured by a simple state-action mapping, and when the reward function of the task is unknown or difficult to specify.

!!! Hint
    One easy way to differentiate the above mentioned domains is as follows, [3]
    
    - In IRL, we recover a Reward function
    - In Apprenticeship Learning, we find a good policy using the recovered reward function from IRL
    - In Behavior cloning, we directly learn the teacher's policy using supervised learning

## Resources

If you are looking for a deep dive into Reinforcement learning, here are some good materials, :ok_hand: 

-  I cannot think of a better free course than [David Silver on Youtube](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-). 
-  In case you prefer books, here is the best of them all - [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). 
-  If you love the API doc way of learning, here is [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html) 


## References

[1] Reinforcement Learning - [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)

[2] Apprenticeship Learning via Inverse Reinforcement Learning - [Paper](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)

[3] Inverse Reinforcement Learning - [Lecture PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/inverseRL.pdf)

