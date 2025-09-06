from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from src.environment.battle_env import BattleEnv

class BattleAgentWrapper(SingleAgentWrapper):

    def __init__(self, env, opponent):
        super().__init__(env, opponent)
    
    def current_battle(self):
        return self.env.current_battle