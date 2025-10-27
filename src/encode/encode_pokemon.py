from poke_env.battle.pokemon import Pokemon
from src.encode import encode_common
from src.utils import pokemon_dict
from typing import Dict
import numpy as np


SIZE_ABILITY = 310
SIZE_BASE_STATS = 6
SIZE_CURRENT_HP = 1
SIZE_ITEM = 88
SIZE_LEVEL = 1

def encode_ability(pokemon: Pokemon) -> np.ndarray:
    ability = pokemon.ability
    array_return = np.zeros(SIZE_ABILITY)
    if ability is None or ability is "":
        return array_return 
    
    ability = ability.replace("'","")
    if ability in pokemon_dict.ABILITY:
        index_ability = pokemon_dict.ABILITY[ability]
        array_return[index_ability] = 1.0
    return array_return

def encode_base_stats(pokemon: Pokemon) -> np.ndarray:
    base_stats = pokemon.base_stats
    array_return = np.zeros(SIZE_BASE_STATS)
    for stat, value in base_stats.items():
        lower_stat = stat.lower()
        if lower_stat in pokemon_dict.BASE_STAT:
            index_stat = pokemon_dict.BASE_STAT[lower_stat]
            array_return[index_stat]=min(value/250.0,1.0)
    return array_return

def encode_boosts(pokemon: Pokemon) -> np.ndarray:
    boosts = pokemon.boosts or {}
    return encode_common.encode_boosts(boosts)

def encode_current_hp(pokemon: Pokemon) -> np.ndarray:
    return np.array([pokemon.current_hp_fraction])

def encode_item(pokemon: Pokemon) -> np.ndarray:
    item = pokemon.item
    array_return = np.zeros(SIZE_ITEM)
    if item is None or item is "":
        return array_return
    item = item.replace("'","")
    if item in pokemon_dict.ITEM:
        index_item = pokemon_dict.ITEM[item]
        array_return[index_item] = 1.0
    return array_return

def encode_level(pokemon: Pokemon) -> np.ndarray:
    return np.array([pokemon.level/100])

def encode_possibile_abilties(pokemon: Pokemon) -> np.ndarray:
    if pokemon.ability is not None:
        return encode_ability(pokemon)

    array_return = np.zeros(SIZE_ABILITY)
    for ability in pokemon.possible_abilities:
        if ability in pokemon_dict.ABILITY:
            index_ability = pokemon_dict.ABILITY[ability]
            array_return[index_ability] = 1.0
    return array_return

def encode_status(pokemon: Pokemon) -> np.ndarray:
    return encode_common.encode_status(pokemon.status)

def encode_type_pokemon(pokemon: Pokemon) -> np.ndarray:
    if pokemon.type_2 is None:
        return encode_common.encode_type(pokemon.type_1, pokemon.type_2)
    else:
        return encode_common.encode_type(pokemon.type_1)

DICT_ENCODE_POKEMON = {
        "ability": (encode_ability, SIZE_ABILITY),
        "base_stats": (encode_base_stats, SIZE_BASE_STATS),
        "boosts": (encode_boosts, encode_common.SIZE_BOOSTS),
        "current_hp": (encode_current_hp, SIZE_CURRENT_HP),
        "item": (encode_item, SIZE_ITEM),
        "level": (encode_level, SIZE_LEVEL),
        "possible_abilities": (encode_possibile_abilties, SIZE_ABILITY),
        "status": (encode_status, encode_common.SIZE_STATUS),
        "type_pokemon": (encode_type_pokemon, encode_common.SIZE_TYPE),
        }
