import torch 
import numpy as np
from poke_env.player.player import Player 
from poke_env.environment.singles_env import SinglesEnv

class PolicyPlayer(Player):
    def __init__(self, policy, calculate):
        self.policy = policy 
        self.calculate = calculate 

    def choose_move(self, battle):
        list_state = [ calc(battle) for calc in self.calculate]
        state = np.concatenate(list_state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = None
        with torch.no_grad():
            action = self.policy_net(state).max(1).indices.view(1,1)
        action = action.item()
        return SinglesEnv.action_to_order(action)