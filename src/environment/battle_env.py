from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle.status import Status
from gymnasium.spaces import Box 
from src.utils import encode_battle
import numpy as np
from typing import Dict, Callable

class BattleEnv(SinglesEnv):

    def __init__(self, 
                encode_f: tuple[Callable, int],
                *, 
                account_configuration1 = None, 
                account_configuration2 = None, 
                avatar = None, 
                battle_format = "gen8randombattle", 
                log_level = None, 
                save_replays = False, 
                server_configuration = LocalhostServerConfiguration, 
                accept_open_team_sheet = False, 
                start_timer_on_battle_start = False, 
                start_listening = True, 
                open_timeout = 10, 
                ping_interval = 20, 
                ping_timeout = 20, 
                team = None, 
                fake = False, 
                strict = True,
                hp_value=1.0,
                opponent_hp_value=1.0,
                fainted_value=1.5,
                opponent_fainted_value=1.5,
                status=0.3,
                opponent_status=0.3,
                victory_value=1,
                turn=0.01,
                number_pokemon=6):
        
        super().__init__(account_configuration1=account_configuration1, 
                        account_configuration2=account_configuration2, 
                        avatar=avatar, 
                        battle_format=battle_format, 
                        log_level=log_level, 
                        save_replays=save_replays, 
                        server_configuration=server_configuration, 
                        accept_open_team_sheet=accept_open_team_sheet, 
                        start_timer_on_battle_start=start_timer_on_battle_start, 
                        start_listening=start_listening, 
                        open_timeout=open_timeout, 
                        ping_interval=ping_interval, 
                        ping_timeout=ping_timeout, 
                        team=team, 
                        fake=fake, 
                        strict=strict)
        
        self.name_agent = account_configuration1.username
        self.hp_value = hp_value
        self.opponent_hp_value = opponent_hp_value
        self.fainted_value = fainted_value
        self.opponent_fainted_value = opponent_fainted_value
        self.status = status
        self.opponent_status = opponent_status
        self.victory_value = victory_value
        self.number_pokemon = number_pokemon
        self.current_battle = None
        self.turn = turn

        size_observation_space = 0
        self.encode = []
        for value in encode_f:
            size_observation_space += value[1]
            self.encode.append(value[0])

        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(size_observation_space,),
                dtype=np.float64) for agent in self.possible_agents
        }


    def step(self, actions):
        return super().step(actions)
    
    def calc_reward(self, battle):
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = 0
        print("team_preview")
        print(battle.teampreview_opponent_team)
        current_value = self.score_state(battle)
        current_value -= self.turn * battle.turn
        if battle.won:
            current_value += self.victory_value
        elif battle.lost:
            current_value -= self.victory_value

        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return
    
    def score_state(self, battle):
        current_value = 0.0
        for mon in battle.team.values():
            if mon.fainted:
                current_value -= self.fainted_value
            else:
                current_value += mon.current_hp_fraction * self.hp_value
                if mon.status is not None:
                    current_value -= self.status

        current_value += (self.number_pokemon - len(battle.team)) * self.hp_value

        for mon in battle.opponent_team.values():
            if mon.fainted:
                current_value += self.opponent_fainted_value
            else:
                current_value -= mon.current_hp_fraction * self.opponent_hp_value
                if mon.status is not None: 
                    current_value += self.opponent_status

        current_value -= (self.number_pokemon - len(battle.opponent_team)) * self.opponent_hp_value
        return current_value

    def embed_battle(self, battle):
        list_return = []
        
        for encode in self.encode:
            list_return.append(encode(battle))

        # turn  = encode_battle.encode_turn(battle)
        # weather = encode_battle.encode_weather(battle)
        # side_condition = encode_battle.encode_side_condition(battle)
        # active_pokemon = encode_battle.encode_active_pokemon(battle)
        # team = encode_battle.encode_team(battle)
        # last_move = encode_battle.encode_last_moves(battle)

        self.current_battle = battle

        return np.concatenate(list_return)

    def reset(self, seed = None, options = None):
        self.list_actions = []
        self._score = []
        return super().reset(seed, options)
