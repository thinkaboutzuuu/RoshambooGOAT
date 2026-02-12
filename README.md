## RoshambooGOAT
RoshambooGOAT is a goated repeated-Roshamboo (Rock-Paper-Scissors) agent that can beat 40/43 benchmark roshamboo agents described in this paper: 

[Lanctot, Marc, et al. "Population-based evaluation in repeated rock-paper-scissors as a benchmark for multiagent reinforcement learning." arXiv preprint arXiv:2303.03196 (2023).](https://arxiv.org/pdf/2303.03196)


### Game Setup:
1. 2 agents plays 1000 rounds roshamboo
   - agent choosing Rock wins if the other agent chooses Scissors, lose if Paper, tie if Rock
   - agent choosing Scissors wins if the other agent chooses Paper, lose if Rock, tie if Scissors
   - agent choosing Paper wins if the other agent chooses Rock, lose if Scissors, tie if Paper
2. an agent beats the other means out of the 1000 games, the agent wins >500 games.
3. agent will have history of the other agents from the previous 20 games.

### Strategy:
1. Mixture of Expert, 7 experts in total, they are:
   - Frequency Based (plays acction that beats most frequent action opponents takes)
   - Opponents Opponent (plays action that beats opponents last move)
   - Hidden Markov-Model (plays action that beats opponents action by markovs model)
   - Mirror Self (plays the same action all the time)
   - Machine Learning (LSTM) (plays action that beats 'predicted' opponents move)
   - Random Play (randomly plays an action under uniform distribution)
   - 9 states MDP (plays action that gives highest utility under a learned mdp (using value iteration))

2. Each Expert has a score playing against the same agent
   - the score decides the action from whick expert should be chosen
   - the score is updated using EMA to accomadate for non-stationarity

3. Final decision is made using Epsilon-greedy strategy
   - take the action the experts suggested with probability (1-epsilon)
   - take a randon action with probability (epsilon)

### Result:
the result is stored in win_rates.csv and win_rates.json, that records the winning rate against each agent.
  - negative score meaning did not beat the agent
   



