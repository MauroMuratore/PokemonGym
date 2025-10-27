from src.encode.encode_battle import DICT_ENCODE_BATTLE 
from src.encode.encode_move import DICT_ENCODE_MOVE
from src.encode.encode_pokemon import DICT_ENCODE_POKEMON
from typing import Dict, Optional, Tuple, Callable

from src.encode.encoder import Encoder

class FactoryEncoder():

    def __init__(self, encoder_dictionary: Optional[Dict[str,Dict[str, tuple[Callable, int]]]]):

        if encoder_dictionary is None:
            self.encoder_dictionary = {
                    "battle": DICT_ENCODE_BATTLE,
                    "move": DICT_ENCODE_BATTLE,
                    "pokemon": DICT_ENCODE_POKEMON
                    }
        else:
            self.encoder_dictionary = encoder_dictionary

    def create_encoder(self, config) -> Encoder:
        list_battle = []
        for encode in config["battle"]:
            assert encode in self.encoder_dictionary["battle"]
            list_battle.append(self.encoder_dictionary["battle"][encode])

        list_pokemon, list_pokemon_move = self._read_encode_pokemon(config, "active_pokemon")
        list_opponent_pokemon, list_opponent_pokemon_move = self._read_encode_pokemon(config, "opponent_active_pokemon")
        list_team, list_team_move = self._read_encode_pokemon(config, "team")
        list_opponent_team, list_opponent_team_move = self._read_encode_pokemon(config, "opponent_team")
        return Encoder(
                list_battle,
                list_pokemon,
                list_pokemon_move,
                list_team,
                list_team_move,
                list_opponent_pokemon,
                list_opponent_pokemon_move,
                list_opponent_team,
                list_opponent_team_move)
    
    def _read_encode_pokemon(self, config, tag):
        list_pokemon = []
        list_pokemon_move = []
        for encode in config[tag]:
            if encode == "moves":
                for encode_move in encode:
                    assert encode_move in self.encoder_dictionary["move"]
                    list_pokemon_move.append(self.encoder_dictionary["move"][encode_move])
            else:
                assert encode in self.encoder_dictionary["pokemon"]
                list_pokemon.append(self.encoder_dictionary["pokemon"][encode])
        
        return list_pokemon, list_pokemon_move
