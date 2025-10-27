from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from src.utils import pokemon_dict
from typing import Dict, Optional
import numpy as np

SIZE_WEATHER = 8
SIZE_SIDE_CONDITION = 11
SIZE_BOOSTS = 7
SIZE_TYPE = 18
SIZE_STATUS = 6

def encode_boosts(boosts: Dict[str, int]) -> np.ndarray:
    array_return = np.zeros(SIZE_BOOSTS)
    if boosts is None:
        return array_return
    for boost, value in boosts.items():
        index_boost = pokemon_dict.BOOST[boost.lower()]
        array_return[index_boost] = min((value + 6.0)/12.0, 1.0)
    return array_return

def encode_side_condition(condition: Optional[SideCondition], value: float = 1.0) -> np.ndarray:
    array_return = np.zeros(SIZE_SIDE_CONDITION)

    if condition in pokemon_dict.STACK_HAZARD:
        index_condition = pokemon_dict.STACK_HAZARD[condition]
        array_return[index_condition] = value 
    elif condition in pokemon_dict.NOT_STACK_HAZARD:
        index_condition = pokemon_dict.NOT_STACK_HAZARD[condition]
        array_return[index_condition] = value
    elif condition in pokemon_dict.EXPIRE_CONDITION:
        index_condition = pokemon_dict.EXPIRE_CONDITION[condition]
        array_return[index_condition] = value

    return array_return

def encode_status(status: Optional[Status]) -> np.ndarray:
    array_return = np.zeros(SIZE_STATUS)
    if status in pokemon_dict.STATUS:
        index_status = pokemon_dict.STATUS[status]
        array_return[index_status]=1.0
    return array_return

def encode_type(type_1: PokemonType, type_2: Optional[PokemonType] = None) -> np.ndarray:
    array_return = np.zeros(SIZE_TYPE)
    if type_1 in pokemon_dict.POKEMON_TYPE:
        index_type_1 = pokemon_dict.POKEMON_TYPE[type_1]
        array_return[index_type_1]=1.0
    if type_2 in pokemon_dict.POKEMON_TYPE:
        index_type_2 = pokemon_dict.POKEMON_TYPE[type_2]
        array_return[index_type_2]=1.0
    return array_return

def encode_weather(weather: Optional[Weather], value: float = 1.0) -> np.ndarray:
    array_return = np.zeros(SIZE_WEATHER)
    if weather in pokemon_dict.WEATHER:
        index_weather = pokemon_dict.WEATHER[weather]
        array_return[index_weather] = value
    return array_return
