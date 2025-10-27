from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition, STACKABLE_CONDITIONS
from src.encode import encode_common
from src.utils import pokemon_dict
from typing import Dict
import numpy as np
import re

from src.utils.encode_battle_old import SIZE_SIDE_CONDITION


SIZE_LAST_MOVE = 22
SIZE_TURN = 1

def encode_last_move(battle: AbstractBattle) -> np.ndarray:
    if battle.turn < 1 or not battle.observations:
        return np.zeros(SIZE_LAST_MOVE)
    
    if "p1" in list(battle.team.keys())[0]:
        prefix_trainer="p1"
        prefix_opponent="p2"
    else:
        prefix_trainer="p2"
        prefix_opponent="p1"

    team_pokemon = [re.sub("p[12]a: ", "", key) for key in list(battle.team.keys())]
    opponent_team_pokemon = [re.sub("p[12]a: ", "", key) for key in list(battle.opponent_team.keys())]


    if hasattr(battle.active_pokemon, "moves"):
        moves = list(battle.active_pokemon.moves.keys())
    else:
        moves = []
    
    if hasattr(battle.opponent_active_pokemon, "moves"):
        opponent_moves = list(battle.opponent_active_pokemon.moves.keys())
    else:
        opponent_moves = []


    last_observation = battle.observations[battle.turn-1]    
    last_move = -1
    opponent_last_move = -1

    for event in last_observation.events:
        if event[1] == "switch":
            if last_move < 0 and prefix_trainer in event[2]:
                pokemon = re.sub("p[12]a", "", event[2])
                if pokemon in team_pokemon:
                    last_move = team_pokemon.index(pokemon)
            if opponent_last_move < 0 and prefix_opponent in event[2]:
                pokemon = re.sub("p[12]a", "", event[2])
                if pokemon in opponent_team_pokemon:
                    opponent_last_move = opponent_team_pokemon.index(pokemon)
        
        elif event[1] == "move":
            move = event[3].lower().replace(" ","")
            if last_move < 0 and prefix_trainer in event[2]:
                if move in moves:
                    last_move = 5 + moves.index(move)
            if opponent_last_move < 0 and prefix_opponent in event[2]:
                if move in opponent_moves:
                    opponent_last_move = 5 + opponent_moves.index(move)

        elif event[1] == "faint":
            if prefix_trainer in event[2]:
                last_move = 10
            if prefix_opponent in event[2]:
                opponent_last_move = 10

    half_size = round(SIZE_LAST_MOVE / 2)
    array_last_move = np.zeros(half_size)
    if last_move >= 0:
        array_last_move[last_move] = 1.0
    opponent_array_last_move = np.zeros(half_size)
    if opponent_last_move >= 0:
        opponent_array_last_move[opponent_last_move] = 1.0
    
    return np.concatenate([
        array_last_move,
        opponent_array_last_move
        ])

def _encode_side_condition(conditions: Dict[SideCondition, int], current_turn: int) -> np.ndarray:
    array_return = np.zeros(encode_common.SIZE_SIDE_CONDITION)

    for condition, value in conditions.items():
        if condition in pokemon_dict.STACK_HAZARD:
            value = value / STACKABLE_CONDITIONS[condition]
        elif condition in pokemon_dict.NOT_STACK_HAZARD:
            value = 1.0
        elif condition in pokemon_dict.EXPIRE_CONDITION:
            value = min((value + 5 - current_turn) / 5, 1.0)
            value = max(value, 0.0)
        array_return += encode_common.encode_side_condition(condition, value)

    return array_return

def encode_side_condition(battle: AbstractBattle) -> np.ndarray:
    current_turn = battle.turn
    array_side_condition = np.zeros(encode_common.SIZE_SIDE_CONDITION)
    array_opponent_side_condition = np.zeros(encode_common.SIZE_SIDE_CONDITION)
    if battle.side_conditions is not None:
        array_side_condition += _encode_side_condition(battle.side_conditions, current_turn)
    if battle.opponent_side_conditions is not None:
        array_opponent_side_condition += _encode_side_condition(battle.opponent_side_conditions, current_turn)

    return np.concatenate([
        array_side_condition,
        array_opponent_side_condition
        ])

def encode_turn(battle: AbstractBattle) -> np.ndarray:
    turn = max(battle.turn, 0) / 100
    turn = min(turn, 1)
    return np.array([turn])

def encode_weather(battle: AbstractBattle) -> np.ndarray:
    array_return = np.zeros(encode_common.SIZE_WEATHER)
    if len(battle.weather) > 0:
        weather, turn = max(battle.weather.items(), key=lambda x: x[1])
        if turn > 0:
            value = min((turn + 5.0 - battle.turn)/5.0, 1)
            value = max(value, 0.0)
            array_return += encode_common.encode_weather(weather, value)
    return array_return 

DICT_ENCODE_BATTLE = {
        "last_move": (encode_last_move, SIZE_LAST_MOVE),
        "side_condition": (encode_side_condition, 2*encode_common.SIZE_SIDE_CONDITION),
        "turn": (encode_turn, SIZE_TURN),
        "weather": (encode_weather, encode_common.SIZE_WEATHER)
        }

