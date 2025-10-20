import re
from typing import Dict
import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle 
from poke_env.battle.side_condition import SideCondition, STACKABLE_CONDITIONS
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.weather import Weather
from poke_env.battle.target import Target
from poke_env.battle.status import Status
import src.utils.pokemon_dict as pokemon_dict


SIZE_TURN = 1
SIZE_WEATHER = 8
SIZE_SIDE_CONDITION = 11
SIZE_BOOST = 7
SIZE_TYPE = 18
SIZE_BASE_STATS = 6
SIZE_HP=1
SIZE_LEVEL=1
SIZE_STATUS = 6
SIZE_MOVE_CATEGORY = 3
SIZE_DATA_MOVE= 14
SIZE_LAST_MOVE = 11
SIZE_ITEM = 88
SIZE_ABILITY = 310

SIZE_MOVE = SIZE_MOVE_CATEGORY + SIZE_BOOST + SIZE_DATA_MOVE\
        + SIZE_TYPE + SIZE_WEATHER + SIZE_SIDE_CONDITION + SIZE_STATUS

SIZE_POKEMON = SIZE_TYPE + SIZE_BASE_STATS + SIZE_ITEM + SIZE_ABILITY +\
        SIZE_HP + SIZE_LEVEL + SIZE_STATUS + 4 * SIZE_MOVE

SIZE_ACTIVE_POKEMON = SIZE_POKEMON + SIZE_BOOST

SIZE_TEAM = 5 * SIZE_POKEMON

SIZE_BATTLE = SIZE_TURN + SIZE_WEATHER + 2* SIZE_SIDE_CONDITION + 2*SIZE_ACTIVE_POKEMON +\
        2*SIZE_TEAM + 2*SIZE_LAST_MOVE


def encode_turn(battle: AbstractBattle) -> np.array:
    """
    Encodes the current battle turn into a normalized numpy array. Returns array of size 1.
    
    Args:
        battle (AbstractBattle): The battle object containing turn information.
        
    Returns:
        np.array: Normalized turn value between 0 and 1 (turn/100), clamped to [0,1] range.
    """
    turn = max(battle.turn, 0) / 100
    turn = min(turn, 1)
    return np.array([turn])

def encode_weather(battle: AbstractBattle) -> np.array:
    """
    Encodes current weather conditions in the battle. Returns array of size SIZE_WEATHER.
    
    Args:
        battle (AbstractBattle): The battle object containing weather information.
        
    Returns:
        np.array: Current weather condition with normalized values based on remaining turns.
    """
    array_return = np.zeros(SIZE_WEATHER)

    if len(battle.weather) > 0:
        weather, turn = max(battle.weather.items(), key=lambda x: x[1])
        if turn > 0:
            value = min((turn + 5.0 - battle.turn)/5.0, 1)
            value = max(value, 0.0)
            array_return = array_return + _encode_array_weather(weather, value)

    return array_return

def _encode_array_weather(weather: Weather, value: float = 1.0) -> np.array:
    array_return = np.zeros(SIZE_WEATHER)
    if weather in pokemon_dict.WEATHER:
        index_weather = pokemon_dict.WEATHER[weather]
        array_return[index_weather] = value
    return array_return



def encode_side_condition(battle: AbstractBattle) -> np.array:
    """
    Encodes side conditions for both sides of the battle. Returns array of size 2*SIZE_SIDE_CONDITION.
    
    Args:
        battle (AbstractBattle): The battle object containing side conditions.
        
    Returns:
        np.array: Concatenated side conditions for player and opponent. Stackable hazards
                 use normalized layer count, non-stackable use binary values (0/1),
                 time-expiring conditions use normalized remaining turns.
    """

    current_turn = battle.turn
    side_condition = _encode_side_condition(battle.side_conditions, current_turn)
    opponent_side_condition =  _encode_side_condition(battle.opponent_side_conditions, current_turn)

    return np.concatenate([
        side_condition,
        opponent_side_condition,
        ], axis=None)

def _encode_side_condition(conditions: Dict[SideCondition, int], current_turn: int) -> np.array:
    array_return = np.zeros(SIZE_SIDE_CONDITION)

    for condition, value in conditions.items():
        if condition in pokemon_dict.STACK_HAZARD:
            value = value / STACKABLE_CONDITIONS[condition]
        elif condition in pokemon_dict.NOT_STACK_HAZARD:
            value = 1.0
        elif condition in pokemon_dict.EXPIRE_CONDITION:
            value = min((value + 5 - current_turn) / 5, 1.0)
            value = max(value, 0.0)
        array_return = array_return + _encode_array_side_condition(condition, value)

    return array_return

def _encode_array_side_condition(condition: SideCondition, value: float = 1.0) -> np.array:
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

def encode_active_pokemon(battle: AbstractBattle) -> np.array:
    """
    Encodes currently active Pokémon for both players. Returns array of size 2 * SIZE_ACTIVE_POKEMON
    
    Args:
        battle (AbstractBattle): The battle object containing active Pokémon.
        
    Returns:
        np.array: Concatenated data for active Pokémon and their stat boosts for both players.
    """

    if hasattr(battle, "active_pokemon"):
        active_pokemon = _encode_pokemon(battle.active_pokemon)    
        if hasattr(battle.active_pokemon, "boosts"):
            boost_active_pokemon = _encode_boost(battle.active_pokemon.boosts)
        else:
            boost_active_pokemon = np.zeros(SIZE_BOOST)
    else:
        active_pokemon = _empty_pokemon()
        boost_active_pokemon = np.zeros(SIZE_BOOST)
    if hasattr(battle, "opponent_active_pokemon"):
        opponent_active_pokemon = _encode_pokemon(battle.opponent_active_pokemon)
        if hasattr(battle.opponent_active_pokemon, "boosts"):
            opponent_boost_active_pokemon = _encode_boost(battle.opponent_active_pokemon.boosts)
        else:
            opponent_boost_active_pokemon = np.zeros(SIZE_BOOST)
    else:
        opponent_active_pokemon = _empty_pokemon()
        opponent_boost_active_pokemon = np.zeros(SIZE_BOOST)

    return np.concatenate([
        active_pokemon,
        boost_active_pokemon,
        opponent_active_pokemon,
        opponent_boost_active_pokemon
        ], axis=None)

def encode_team(battle: AbstractBattle) -> np.array:
    """
    Encodes both players' teams excluding active Pokémon. Returns array of size 2 * SIZE_TEAM.
    
    Args:
        battle (AbstractBattle): The battle object containing team information.
        
    Returns:
        np.array: Concatenated reserve Pokémon data for both teams, padded to 5 Pokémon each.
    """
    list_team = battle.team.copy()
    team_pokemon = [ _encode_pokemon(pokemon) \
            for _, pokemon in list_team.items() if not pokemon.active]
    list_opponent_team = battle.opponent_team.copy()
    opponent_team_pokemon = [ _encode_pokemon(pokemon) \
            for _, pokemon in list_opponent_team.items() if not pokemon.active]

    for _ in range(5 - len(team_pokemon)):
        team_pokemon.append(_empty_pokemon())
    
    for _ in range(5 - len(opponent_team_pokemon)):
        opponent_team_pokemon.append(_empty_pokemon())

    return np.concatenate([
        *team_pokemon,
        *opponent_team_pokemon
        ], axis=None)

def _encode_boost(boosts: Dict[str, int]) -> np.array:
    array_return = np.zeros(SIZE_BOOST)
    if boosts is None:
        return array_return
    for boost, value in boosts.items():
        index_boost = pokemon_dict.BOOST[boost.lower()]
        array_return[index_boost] = min((value + 6.0)/12.0, 1.0)
    return array_return

def _encode_pokemon(pokemon: Pokemon) -> np.array:
    if hasattr(pokemon, "type_1"):
        if hasattr(pokemon, "type_2"):
            pokemon_type = _encode_type(pokemon.type_1, pokemon.type_2)
        else:
            pokemon_type = _encode_type(pokemon.type_1)
    else:
        pokemon_type = np.zeros(SIZE_TYPE)

    if hasattr(pokemon, "base_stats"):
        pokemon_base_stats = _encode_base_stats(pokemon.base_stats)
    else:
        pokemon_base_stats = np.zeros(SIZE_BASE_STATS)

    if hasattr(pokemon, "current_hp") and hasattr(pokemon, "max_hp"):
        pokemon_hp = _encode_hp(pokemon.current_hp, pokemon.max_hp)
    else:
        pokemon_hp = np.zeros(SIZE_HP)

    if hasattr(pokemon, "level"):
        pokemon_level = np.array([pokemon.level/100])
    else:
        pokemon_level = np.zeros(SIZE_LEVEL)

    if hasattr(pokemon, "status"):
        pokemon_status = _encode_status(pokemon.status)
    else:
        pokemon_status = np.zeros(SIZE_STATUS)

    if hasattr(pokemon, "item"):
        pokemon_item = _encode_item(pokemon.item)
    else: 
        pokemon_item = np.zeros(SIZE_ITEM)

    if hasattr(pokemon, "ability"):
        pokemon_ability = _encode_ability(pokemon.ability)
    else:
        pokemon_ability = np.zeros(SIZE_ABILITY)

    list_moves = []
    if hasattr(pokemon, "moves"):
        for _, move in pokemon.moves.items():
            list_moves.append(_encode_move(move))

    if len(list_moves) < 4:
        for _ in range(4-len(list_moves)):
            list_moves.append(_empty_move())

    return np.concatenate([
        pokemon_type,
        pokemon_base_stats,
        pokemon_hp,
        pokemon_level,
        pokemon_status,
        pokemon_item,
        pokemon_ability,
        *list_moves
        ], axis=None)

def _empty_pokemon() -> np.array:
    return np.zeros(SIZE_POKEMON)

def _encode_type(type_1: PokemonType, type_2: PokemonType = None) -> np.array:
    array_return = np.zeros(SIZE_TYPE)
    if type_1 in pokemon_dict.POKEMON_TYPE:
        index_type_1 = pokemon_dict.POKEMON_TYPE[type_1]
        array_return[index_type_1]=1.0
    if type_2 in pokemon_dict.POKEMON_TYPE:
        index_type_2 = pokemon_dict.POKEMON_TYPE[type_2]
        array_return[index_type_2]=1.0
    return array_return

def _encode_base_stats(base_stats: Dict[str, int]) -> np.array:
    array_return = np.zeros(SIZE_BASE_STATS)
    for stat, value in base_stats.items():
        lower_stat = stat.lower()
        if lower_stat in pokemon_dict.BASE_STAT:
            index_stat = pokemon_dict.BASE_STAT[lower_stat]
            array_return[index_stat]=min(value/250.0,1.0)
    return array_return

def _encode_hp(current_hp:int, max_hp:int) -> np.array:
    if max_hp > 0:
        pokemon_hp = current_hp / max_hp
    else:
        pokemon_hp = current_hp
    return np.array([pokemon_hp])

def _encode_status(status: Status) -> np.array:
    array_return = np.zeros(SIZE_STATUS)
    if status in pokemon_dict.STATUS:
        index_status = pokemon_dict.STATUS[status]
        array_return[index_status]=1.0
    return array_return

def _encode_item(item: str) -> np.array:
    array_return = np.zeros(SIZE_ITEM)
    if item is None:
        return array_return
    item = item.lower().replace("'","")
    if item in pokemon_dict.ITEM:
        index_item = pokemon_dict.ITEM[item]
        array_return[index_item]=1.0
    return array_return

def _encode_ability(ability: str) -> np.array:
    array_return = np.zeros(SIZE_ABILITY)
    if ability is None:
        return array_return
    ability = ability.lower().replace("'","")
    if ability in pokemon_dict.ABILITY:
        index_ability = pokemon_dict.ABILITY[ability]
        array_return[index_ability]=1.0
    return array_return

def _encode_move(move: Move) -> np.array:
    move_category = _encode_move_category(move.category)
    boosts = _encode_boost(move.boosts)
    status = _encode_status(move.status)
    type_move = _encode_type(move.type)
    weather = _encode_array_weather(move.weather)
    side_condition = _encode_array_side_condition(move.side_condition)
    
    remaining_move_array = np.zeros(SIZE_DATA_MOVE)
    remaining_move_array[0] = move.accuracy
    remaining_move_array[1] = min(move.base_power/200.0, 1.0)
    remaining_move_array[2] = move.crit_ratio / 6.0
    remaining_move_array[3] = move.current_pp / move.max_pp
    remaining_move_array[4] = move.drain
    remaining_move_array[5] = min(move.heal, 1.0)
    remaining_move_array[6] = min(move.n_hit[0] / 10.0, 1.0)
    remaining_move_array[7] = min(move.n_hit[1] / 10.0, 1.0)
    remaining_move_array[8] = 1 if move.is_protect_move else 0
    remaining_move_array[9] = (move.priority +6.0)/12.0
    remaining_move_array[10] = move.recoil
    if move.self_destruct:
        remaining_move_array[11] = 1.0
    if move.self_switch:
        remaining_move_array[12] = 1.0
    if move.target is Target.SELF:
        remaining_move_array[13] = 1.0

    return np.concatenate([
        move_category,
        boosts,
        status,
        type_move,
        weather,
        side_condition,
        remaining_move_array
        ])

def _empty_move() -> np.array:
    return np.zeros(SIZE_MOVE)

def _encode_move_category(move_category: MoveCategory) -> np.array:
    array_return = np.zeros(SIZE_MOVE_CATEGORY)
    if move_category in pokemon_dict.MOVE_CATEGORY:
        index_move_category = pokemon_dict.MOVE_CATEGORY[move_category]
        array_return[index_move_category] = 1
    return array_return

def encode_last_moves(battle: AbstractBattle) -> np.array:
    if battle.turn < 1 or not battle.observations:
        return np.zeros(2*SIZE_LAST_MOVE)
    
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

    array_last_move = np.zeros(SIZE_LAST_MOVE)
    if last_move >= 0:
        array_last_move[last_move] = 1.0
    opponent_array_last_move = np.zeros(SIZE_LAST_MOVE)
    if opponent_last_move >= 0:
        opponent_array_last_move[opponent_last_move] = 1.0
    
    return np.concatenate([
        array_last_move,
        opponent_array_last_move
        ])


DICTIONARY_ENCODE= {
    "turn": (encode_turn, SIZE_TURN),
    "weather": (encode_weather, SIZE_WEATHER),
    "side_condition": (encode_side_condition, 2*SIZE_SIDE_CONDITION),
    "active_pokemon": (encode_active_pokemon, 2*SIZE_ACTIVE_POKEMON),
    "team": (encode_team, 2*SIZE_TEAM),
    "last_move": (encode_last_moves, 2*SIZE_LAST_MOVE)
}