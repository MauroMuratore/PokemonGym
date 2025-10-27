from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory 
from poke_env.battle.target import Target
from src.encode import encode_common
from src.utils import pokemon_dict
import numpy as np

SIZE_MOVE_CATEGORY = 3
SIZE_DATA_MOVE= 14

def encode_boosts(move: Move) -> np.ndarray:
    boosts = move.boosts or {}
    return encode_common.encode_boosts(boosts)

def encode_category(move: Move) -> np.ndarray:
    move_category = move.category 
    array_return = np.zeros(SIZE_MOVE_CATEGORY)
    if move_category in pokemon_dict.MOVE_CATEGORY:
        index_move_category = pokemon_dict.MOVE_CATEGORY[move_category]
        array_return[index_move_category] = 1
    return array_return

def encode_data_move(move: Move) -> np.ndarray:
    data_move = np.zeros(SIZE_DATA_MOVE)
    data_move[0] = move.accuracy
    data_move[1] = min(move.base_power/200.0, 1.0)
    data_move[2] = move.crit_ratio / 6.0
    data_move[3] = move.current_pp / move.max_pp
    data_move[4] = move.drain
    data_move[5] = min(move.heal, 1.0)
    data_move[6] = min(move.n_hit[0] / 10.0, 1.0)
    data_move[7] = min(move.n_hit[1] / 10.0, 1.0)
    data_move[8] = 1 if move.is_protect_move else 0
    data_move[9] = (move.priority +6.0)/12.0
    data_move[10] = move.recoil
    if move.self_destruct:
        data_move[11] = 1.0
    if move.self_switch:
        data_move[12] = 1.0
    if move.target is Target.SELF:
        data_move[13] = 1.0
    return data_move

def encode_side_condition(move: Move) -> np.ndarray:
    return encode_common.encode_side_condition(move.side_condition)

def encode_status(move: Move) -> np.ndarray:
    return encode_common.encode_status(move.status)

def encode_type_pokemon(move: Move) -> np.ndarray:
    return encode_common.encode_type(move.type)

def encode_weather(move: Move) -> np.ndarray:
    return encode_common.encode_weather(move.weather)

DICT_ENCODE_MOVE = {
        "boosts": (encode_boosts, encode_common.SIZE_BOOSTS),
        "category": (encode_category, SIZE_DATA_MOVE),
        "data_move": (encode_data_move, SIZE_DATA_MOVE),
        "side_condition": (encode_side_condition, encode_common.SIZE_SIDE_CONDITION),
        "status": (encode_status, encode_common.Status),
        "type_pokemon": (encode_type_pokemon, encode_common.SIZE_TYPE),
        "weather": (encode_weather, encode_common.SIZE_WEATHER)
        }
