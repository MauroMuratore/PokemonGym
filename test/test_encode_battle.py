from unittest.mock import Mock
import numpy as np
from poke_env.battle.weather import Weather
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.pokemon_type import PokemonType
import src.utils.encode_battle as encode_battle
import src.utils.pokemon_dict as pokemon_dict

def test_turn():
    print("Test turn: ", end="")

    mock_battle = Mock()
    mock_battle.turn = 0
    result = np.zeros(1)
    test_t_0 = encode_battle.encode_turn(mock_battle)
    np.testing.assert_array_equal(test_t_0, result)
    
    mock_battle.turn = -1
    test_t_1 = encode_battle.encode_turn(mock_battle)
    np.testing.assert_array_equal(test_t_1, result)

    mock_battle.turn = 50
    test_t_2 = encode_battle.encode_turn(mock_battle)
    result = np.array([0.5])
    np.testing.assert_array_equal(test_t_2, result)

    mock_battle.turn = 150
    test_t_3 = encode_battle.encode_turn(mock_battle)
    result = np.array([1.0])
    np.testing.assert_array_equal(test_t_3, result)

    print("Successful")
    

def test_weather():
    print("Test weather: ")
    weather_dict = pokemon_dict.WEATHER
    mock_battle = Mock()
    mock_battle.turn = 10
    mock_battle.weather = {}

    print("\t- Test empty: ", end="")
    result = np.zeros(encode_battle.SIZE_WEATHER)
    test_w_0 = encode_battle.encode_weather(mock_battle)
    np.testing.assert_array_equal(test_w_0, result)
    print("Successful")

    
    print("\t- Test expire raindance: ", end="")
    mock_battle.weather[Weather.RAINDANCE] = 4
    index_raindance = weather_dict[Weather.RAINDANCE]
    test_w_1 = encode_battle.encode_weather(mock_battle)
    np.testing.assert_array_equal(test_w_1, result)
    print("Successful")

    print("\t- Test current raindance: ", end="")
    mock_battle.weather[Weather.RAINDANCE] = 8
    result[index_raindance] = 3.0 /5.0
    test_w_2 = encode_battle.encode_weather(mock_battle)
    np.testing.assert_array_equal(test_w_2, result)
    print("Successful")

    print("\t- Test old sunnyday: ", end="")
    mock_battle.weather[Weather.SUNNYDAY] = 7
    test_w_3 = encode_battle.encode_weather(mock_battle)
    np.testing.assert_array_equal(test_w_3, result)
    print("Successful")

    print("\t- Test current hail: ", end="")
    mock_battle.weather[Weather.HAIL] = 9
    index_hail = weather_dict[Weather.HAIL]
    result[index_raindance] = 0.0
    result[index_hail] = 4.0/5.0
    test_w_4 = encode_battle.encode_weather(mock_battle)
    np.testing.assert_array_equal(test_w_4, result)
    print("Successful")
    print("Successful")

def test_side_condition():
    mock_battle = Mock()
    mock_battle.turn = 10
    side_conditions = {}
    opponent_side_conditions = {}
    mock_battle.side_conditions = side_conditions
    mock_battle.opponent_side_conditions = opponent_side_conditions
    size_side_condition = encode_battle.SIZE_SIDE_CONDITION
    print("Test side condition")
    
    print("\t- Test empty side condition: ", end="")
    result = np.zeros(2*size_side_condition)
    test_sc_0 = encode_battle.encode_side_condition(mock_battle)
    np.testing.assert_array_equal(test_sc_0, result)
    print("Successful")
    
    print("\t- Test spikes: ", end="")
    side_conditions[SideCondition.SPIKES] = 2.0
    mock_battle.side_conditions = side_conditions
    index_spikes = pokemon_dict.STACK_HAZARD[SideCondition.SPIKES]
    result[index_spikes] = 2.0 / 3.0
    test_sc_1 = encode_battle.encode_side_condition(mock_battle)
    np.testing.assert_array_equal(test_sc_1, result)
    print("Successful")

    print("\t- Test stealth rock: ", end="")
    side_conditions[SideCondition.STEALTH_ROCK] = 8
    mock_battle.side_conditions = side_conditions
    index_stealth_rock = pokemon_dict.NOT_STACK_HAZARD[SideCondition.STEALTH_ROCK]
    result[index_stealth_rock] = 1.0
    test_sc_2 = encode_battle.encode_side_condition(mock_battle)
    np.testing.assert_array_equal(test_sc_2, result)
    print("Successful")

    print("\t- Test opponent expire reflect: ", end="")
    opponent_side_conditions[SideCondition.REFLECT] = 4
    mock_battle.opponent_side_conditions = opponent_side_conditions
    test_sc_3 = encode_battle.encode_side_condition(mock_battle)
    np.testing.assert_array_equal(test_sc_3, result)
    print("Successful")

    print("\t- Test opponent current tailwind: ", end="")
    opponent_side_conditions[SideCondition.TAILWIND] = 8 
    mock_battle.opponent_side_conditions = opponent_side_conditions
    index_tailwind = size_side_condition + \
            pokemon_dict.EXPIRE_CONDITION[SideCondition.TAILWIND]
    result[index_tailwind] = 3.0 / 5.0
    test_sc_4 = encode_battle.encode_side_condition(mock_battle)
    np.testing.assert_array_equal(test_sc_4, result)
    print("Successful")
    print("Successful")

def test_pokemon():
    print("Test pokemon: ")
    mock_pokemon = Mock(spec=[])
    offset = 0
    print("\t- Test empty pokemon: ", end="")
    result = np.zeros(encode_battle.SIZE_POKEMON)
    test_pk_0 = encode_battle._encode_pokemon(mock_pokemon) 
    np.testing.assert_array_equal(test_pk_0, result)
    print("Successful")

    print("\t- Test type 1 pokemon: ", end="")
    mock_pokemon.type_1 = PokemonType.NORMAL
    index_type_1 = pokemon_dict.POKEMON_TYPE[PokemonType.NORMAL]
    result[index_type_1] = 1
    test_pk_1 = encode_battle._encode_pokemon(mock_pokemon)
    np.testing.assert_array_equal(test_pk_1, result)
    print("Successful")

    print("\t- Test type 2 pokemon: ", end="")
    mock_pokemon.type_2 = PokemonType.GHOST
    index_type_2 = pokemon_dict.POKEMON_TYPE[PokemonType.GHOST]
    result[index_type_2] = 1
    test_pk_2 = encode_battle._encode_pokemon(mock_pokemon)
    np.testing.assert_array_equal(test_pk_2, result)
    print("Successful")

    offset = offset + encode_battle.SIZE_TYPE

    print("\t- Test base_stats pokemon: ", end="")

    mock_pokemon.base_stats = {
        "atk": 25,
        "DEF": 500,
        "spK": 125,
    }
    index_atk = offset + pokemon_dict.BASE_STAT["atk"]
    value_atk = 25.0/250.0
    index_def = offset + pokemon_dict.BASE_STAT["def"]
    value_def = 1
    result[index_atk] = value_atk
    result[index_def] = value_def
    test_pk_3 = encode_battle._encode_pokemon(mock_pokemon)
    np.testing.assert_array_equal(test_pk_3, result)
    print("Successful")

    print("\t- Test current hp pokemon: ", end="")
    print("Successful")
    print("\t- Test level pokemon: ", end="")
    print("Successful")
    print("\t- Test status pokemon: ", end="")
    print("Successful") 
    print("\t- Test moves pokemon: ", end="")
    print("Successful")

    print("Successful")




if __name__ == "__main__":
    test_turn()
    test_weather()
    test_side_condition()
    test_pokemon()
