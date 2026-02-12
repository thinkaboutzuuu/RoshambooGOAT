import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import pyspiel


class RPSLSTM(nn.Module):
    def __init__(self, vocab_size=3, emb_dim=8, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # logits for R,P,S

    def forward(self, x, hidden=None):
        # x: (batch, T) of ints 0/1/2
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)  # out: (batch, T, hidden)
        last = out[:, -1, :]
        logits = self.fc(last)                 # (batch, T, 3)
        return logits, hidden
    

class MyAgent21084645(rl_agent.AbstractAgent):
    """
    Greenberg-lite ensemble agent for repeated RPS:

    - Maintains several simple "expert" strategies (predictors + best-response).
    - Tracks a score for each expert based on how well it *would have* done.
    - At each step, picks the action of the best-scoring expert (with some
      epsilon exploration).
    """

    def __init__(self,
                 player_id,
                 num_actions=3,
                 score_alpha=0.5,      
                 epsilon_action=0.1, 
                 epsilon_expert=0.05, 
                 name="ensemble_agent",
                 lstm_model_path="rps_lstm.pt",
                 lstm_seq_len=200, 
                 gamma = 0.95,
                 mdp_interval = 50):
        super().__init__(player_id=player_id, name=name)
        assert num_actions == 3, "RRPS uses 3 actions (R,P,S)."
        self._player_id = player_id
        self._num_actions = num_actions

        self._score_alpha = score_alpha
        self._eps_action = epsilon_action
        self._eps_expert = epsilon_expert

        self._payoff = np.array([
            [ 0, -1,  1],  
            [ 1,  0, -1],  
            [-1,  1,  0],  
        ], dtype=float)

        # Number of experts we'll define
        self._num_experts = 7
        self._lstm_seq_len = lstm_seq_len
        self._device = torch.device("mps")
        self._lstm_model = RPSLSTM(vocab_size=4, emb_dim=16, hidden_size=64).to(self._device)
        state_dict = torch.load(lstm_model_path, map_location=self._device)
        self._lstm_model.load_state_dict(state_dict)
        self._lstm_model.eval()
        self._lstm_hidden = None

        self._mdp_num_states = self._num_actions*self._num_actions + 1
        self._mdp_start_state = self._mdp_num_states-1
        self._mdp_gamma = gamma
        self._mdp_plan_interval = mdp_interval

        self._mdp_N_sa = np.zeros((self._mdp_num_states, self._num_actions), dtype=np.float64)
        self._mdp_R_sa_sum = np.zeros((self._mdp_num_states, self._num_actions), dtype=np.float64)
        self._mdp_N_sas = np.zeros(
            (self._mdp_num_states, self._num_actions, self._mdp_num_states), dtype=np.float64
        )

        # Value function and policy
        self._mdp_V = np.zeros(self._mdp_num_states, dtype=np.float64)
        self._mdp_policy = -np.ones(self._mdp_num_states, dtype=int)

        self._mdp_prev_state = None
        self._mdp_prev_action = None
        self._mdp_steps_since_plan = 0

        self.restart()

    # ------------ Episode reset ------------

    def restart(self):
        # Opponent statistics
        self._opp_counts = np.ones(self._num_actions, dtype=float)  # smoothed counts
        self._trans_counts = np.ones((self._num_actions, self._num_actions),
                                     dtype=float)  # P(opp_t | opp_{t-1})

        self._last_opp_action = None
        self._last_my_action = None

        # Expert scores and last actions they recommended
        self._expert_scores = np.zeros(self._num_experts, dtype=float)
        self._last_expert_actions = [None] * self._num_experts

        # Track how much of history we've processed (if we use it later)
        self._last_history_len = 0

        self._opp_hist = []
        self._lstm_hidden = None

        self._mdp_prev_state = None
        self._mdp_prev_action = None
        self._mdp_steps_since_plan = 0

    # ------------ Utility helpers ------------

    def _beat(self, move):
        #Return the action that beats 'move'
        if move is None:
            return np.random.randint(self._num_actions)
        return (move + 1) % self._num_actions

    def _update_opp_stats(self, state):
        #Update opponent statistics based on game history.
        history = state.history()
        if len(history) == 0:
            return None

        action = history[-1]
        # Update counts
        self._opp_counts[action] += 1
        if self._last_opp_action is not None:
            self._trans_counts[self._last_opp_action, action] += 1
        self._last_opp_action = action
        self._opp_hist.append(action)
        return action
    
    def _lstm_predict_opp(self):
        if len(self._opp_hist) == 0:
            return None

        # take last seq_len moves (pad/truncate)
        seq = self._opp_hist[-self._lstm_seq_len:]
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=self._device)
        seq_tensor = seq_tensor.unsqueeze(0)  # shape (1, T)

        with torch.no_grad():
            # assume model returns (logits, hidden)
            logits, self._lstm_hidden = self._lstm_model(seq_tensor, self._lstm_hidden)
            # use last time step logits
            # last_logits = logits[:, -1, :]      # shape (1, 3)
            last_logits = logits      # shape (1, 3)

            probs = F.softmax(last_logits, dim=-1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        return pred
    
    def _mdp_current_state(self):
        """Encode current MDP state from last joint action."""
        if self._last_my_action is None or self._last_opp_action is None:
            return self._mdp_start_state
        return self._last_my_action * self._num_actions + self._last_opp_action

    def _mdp_observe_transition(self, reward, curr_state):
        """Update empirical MDP using previous (s,a) and current state."""
        if self._mdp_prev_state is None or self._mdp_prev_action is None:
            return
        s = self._mdp_prev_state
        a = self._mdp_prev_action
        s_next = curr_state

        self._mdp_N_sa[s, a] += 1.0
        self._mdp_R_sa_sum[s, a] += reward
        self._mdp_N_sas[s, a, s_next] += 1.0

        self._mdp_steps_since_plan += 1
        if self._mdp_steps_since_plan >= self._mdp_plan_interval:
            self._mdp_plan_policy()
            self._mdp_steps_since_plan = 0

    def _mdp_plan_policy(self, max_iters=100, tol=1e-4):
        """Value iteration on empirical MDP; update V and π(s)."""
        V = self._mdp_V.copy()
        S = self._mdp_num_states
        A = self._num_actions
        gamma = self._mdp_gamma

        for _ in range(max_iters):
            delta = 0.0
            for s in range(S):
                best_val = None
                for a in range(A):
                    n_sa = self._mdp_N_sa[s, a]
                    if n_sa <= 0:
                        continue
                    r_hat = self._mdp_R_sa_sum[s, a] / n_sa
                    p_hat = self._mdp_N_sas[s, a] / n_sa  # shape (S,)
                    val = r_hat + gamma * np.dot(p_hat, V)
                    if (best_val is None) or (val > best_val):
                        best_val = val
                if best_val is None:
                    best_val = 0.0
                delta = max(delta, abs(best_val - V[s]))
                V[s] = best_val
            if delta < tol:
                break

        # Extract greedy policy
        policy = -np.ones(S, dtype=int)
        for s in range(S):
            best_val = None
            best_a = 0
            for a in range(A):
                n_sa = self._mdp_N_sa[s, a]
                if n_sa <= 0:
                    continue
                r_hat = self._mdp_R_sa_sum[s, a] / n_sa
                p_hat = self._mdp_N_sas[s, a] / n_sa
                val = r_hat + gamma * np.dot(p_hat, V)
                if (best_val is None) or (val > best_val):
                    best_val = val
                    best_a = a
            if best_val is not None:
                policy[s] = best_a

        self._mdp_V = V
        self._mdp_policy = policy

    # ------------ Expert strategies ------------

    def _expert_actions(self):
        #Compute each expert's recommended action for step.
        actions = []

        # Expert 0: Frequency-based best response
        # Predict opp's most frequent move overall.
        opp_probs = self._opp_counts / np.sum(self._opp_counts)
        pred0 = int(np.argmax(opp_probs))
        actions.append(self._beat(pred0))

        # Expert 1: Last-opponent-move best response
        # Predict they repeat their last move.
        actions.append(self._beat(self._last_opp_action))

        # Expert 2: Markov(1) best response
        # Predict based on last opp move -> next opp move.
        if self._last_opp_action is not None:
            row = self._trans_counts[self._last_opp_action]
            pred2 = int(np.argmax(row))
            actions.append(self._beat(pred2))
        else:
            # fallback to frequency BR
            actions.append(self._beat(pred0))

        # Expert 3: Mirror-me assumption
        # Predict opp plays what I played last time.
        actions.append(self._beat(self._last_my_action))

        # Expert 4: ML (LSTM)
        pred_lstm = self._lstm_predict_opp()
        if pred_lstm is not None:
            actions.append(self._beat(pred_lstm))
        else:
            # fallback: behave like frequency BR
            actions.append(self._beat(pred0))

        # Expert 5: random bullshit
        import random
        actions.append(random.randint(0, 2))

        # Expert 6: MDP expert (greedy from π(s))
        curr_state = self._mdp_current_state()
        if 0 <= curr_state < self._mdp_num_states and self._mdp_policy[curr_state] != -1:
            mdp_action = int(self._mdp_policy[curr_state])
        else:
            # fallback: same as freq-BR
            mdp_action = self._beat(pred0)
        actions.append(mdp_action)

        return actions

    # ------------ Expert scoring ------------

    def _update_expert_scores(self, last_opp_action):
        """
        Update each expert's score based on what happened in the *previous* round.

        We use the payoff matrix: if expert i had played its suggested action,
        what reward would it have gotten vs last_opp_action?
        """
        if last_opp_action is None:
            return  # nothing to update on the first move

        for i in range(self._num_experts):
            a_i = self._last_expert_actions[i]
            if a_i is None:
                continue
            payoff_i = self._payoff[a_i, last_opp_action]
            # Exponential moving average of payoff
            self._expert_scores[i] = (
                (1 - self._score_alpha) * self._expert_scores[i]
                + self._score_alpha * payoff_i
            )

    # ------------ Main step ------------

    def step(self, time_step, is_evaluation=False):
        # Terminal: return dummy
        if time_step.last():
            probs = np.ones(self._num_actions) / self._num_actions
            return rl_agent.StepOutput(action=0, probs=probs)

        # Deserialize game state
        game, state = pyspiel.deserialize_game_and_state(
            time_step.observations["serialized_state"])

        # 1) Update opponent stats (includes last_opp_action)
        last_opp_action = self._update_opp_stats(state)

        curr_mdp_state = self._mdp_current_state()
        reward = None
        if time_step.rewards is not None:
            reward = time_step.rewards[self._player_id]
        elif self._last_my_action is not None and last_opp_action is not None:
            reward = self._payoff[self._last_my_action, last_opp_action]
        if reward is not None:
            self._mdp_observe_transition(reward, curr_mdp_state)

        # 2) Update expert scores based on previous round outcome
        # We don't need time_step.rewards here; we recompute payoff from matrix.
        self._update_expert_scores(last_opp_action)

        # 3) Each expert proposes an action for THIS round
        expert_actions = self._expert_actions()
        self._last_expert_actions = list(expert_actions)  # store for next scoring

        # 4) Choose which expert to follow (epsilon-greedy over expert scores)
        if np.random.rand() < self._eps_expert:
            chosen_expert = np.random.randint(self._num_experts)
        else:
            chosen_expert = int(np.argmax(self._expert_scores))

        chosen_action = expert_actions[chosen_expert]

        # 5) Add action-level randomness for robustness
        if np.random.rand() < self._eps_action:
            action = np.random.randint(self._num_actions)
        else:
            action = chosen_action

        # Save my last action for next round's mirror expert
        self._last_my_action = action

        self._mdp_prev_state = curr_mdp_state
        self._mdp_prev_action = action

        # 6) Build probability distribution (mostly on chosen_action)
        probs = np.ones(self._num_actions) * (self._eps_action / self._num_actions)
        probs[chosen_action] += 1.0 - self._eps_action

        return rl_agent.StepOutput(action=action, probs=probs)