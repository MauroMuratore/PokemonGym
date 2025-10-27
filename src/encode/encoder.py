from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from typing import List, Callable, Dict
import numpy as np
from poke_env.battle.pokemon import Pokemon

class Encoder():

    def __init__(
            self,
            list_encode_battle: List[tuple[Callable, int]],
            list_encode_active_pokemon: List[tuple[Callable, int]],
            list_encode_active_pokemon_move: List[tuple[Callable, int]],
            list_encode_team: List[tuple[Callable, int]],
            list_encode_team_move: List[tuple[Callable, int]],
            list_encode_opponent_active_pokemon: List[tuple[Callable, int]],
            list_encode_opponent_active_pokemon_move: List[tuple[Callable, int]],
            list_encode_opponent_team: List[tuple[Callable, int]],
            list_encode_opponent_team_move: List[tuple[Callable, int]],
            ):

        # ENCODE BATTLE
        self.list_encode_battle : List[Callable] = [ encode[0] for encode in list_encode_battle]
        self.size_battle = 0
        self.size_battle += sum([encode[1] for encode in list_encode_battle])
        # ENCODE ACTIVE POKEMON
        self.list_encode_active_pokemon : List[Callable] = [ encode[0] for encode in list_encode_active_pokemon] 
        self.size_active_pokemon =0
        self.size_active_pokemon += sum([encode[1] for encode in list_encode_active_pokemon])
        # ENCODE ACTIVE POKEMON MOVE
        self.list_encode_active_pokemon_move : List[Callable] = [ encode[0] for encode in list_encode_active_pokemon_move] 
        self.size_active_pokemon_move = 0 
        self.size_active_pokemon_move += sum([encode[1] for encode in list_encode_active_pokemon_move])
        # ENCODE TEAM
        self.list_encode_team : List[Callable] = [encode[0] for encode in list_encode_team]
        self.size_team = 0
        self.size_team += sum([encode[1] for encode in list_encode_team])
        # ENCODE TEAM MOVE
        self.list_encode_team_move : List[Callable] = [encode[0] for encode in list_encode_team_move]
        self.size_team_move = 0
        self.size_team_move += sum([encode[1] for encode in list_encode_team_move])
        # ENCODE OPPONENT ACTIVE POKEMON 
        self.list_encode_opponent_active_pokemon : List[Callable] = [ encode[0] for encode in list_encode_opponent_active_pokemon] 
        self.size_opponent_active_pokemon=0
        self.size_opponent_active_pokemon += sum([encode[1] for encode in list_encode_opponent_active_pokemon])
        # ENCODE OPPONENT ACTIVE POKEMON MOVE 
        self.list_encode_opponent_active_pokemon_move : List[Callable] = [ encode[0] for encode in list_encode_opponent_active_pokemon_move] 
        self.size_opponent_active_pokemon_move = 0
        self.size_opponent_active_pokemon_move += sum([encode[1] for encode in list_encode_opponent_active_pokemon_move])
        # ENCODE OPPONENT TEAM
        self.list_encode_opponent_team : List[Callable] = [encode[0] for encode in list_encode_opponent_team]
        self.size_opponent_team =0 
        self.size_opponent_team += sum([encode[1] for encode in list_encode_opponent_team])
        # ENCODE OPPONENT TEAM MOVE
        self.list_encode_opponent_team_move : List[Callable] = [encode[0] for encode in list_encode_opponent_team_move]
        self.size_opponent_team_move =0
        self.size_opponent_team_move += sum([encode[1] for encode in list_encode_opponent_team_move])

        self.observation_space = self.size_battle +\
            self.size_active_pokemon + 4*self.size_active_pokemon_move +\
            6*(self.size_team + 4*self.size_team_move) +\
            self.size_opponent_active_pokemon + 4*self.size_opponent_active_pokemon_move +\
            6*(self.size_opponent_team + 4*self.size_opponent_team_move)

    def encode(self, battle: AbstractBattle) -> np.ndarray: 
        encoded_return = []
        encoded_return.append(self._encode_battle(battle))
        encoded_return.append(self._encode_pokemon(battle.active_pokemon, self.list_encode_active_pokemon))
        encoded_return.append(self._encode_moves(battle.active_pokemon.moves, self.list_encode_active_pokemon_move, self.size_active_pokemon_move))
        for _, pokemon in battle.team.items():
            encoded_return.append(self._encode_pokemon(pokemon, self.list_encode_team))
            encoded_return.append(self._encode_moves(pokemon.moves, self.list_encode_team_move, self.size_team_move))
        for i in range(0, 6-len(battle.team)):
            encoded_return.append(np.zeros(self.size_team))
            encoded_return.append(np.zeros(4*self.size_team_move))
        encoded_return.append(self._encode_pokemon(battle.opponent_active_pokemon, self.list_encode_opponent_active_pokemon))
        encoded_return.append(self._encode_moves(battle.opponent_active_pokemon.moves, self.list_encode_opponent_active_pokemon_move, self.size_opponent_active_pokemon_move))
        for _, pokemon in battle.opponent_team.items():
            encoded_return.append(self._encode_pokemon(pokemon, self.list_encode_opponent_team))
            encoded_return.append(self._encode_moves(pokemon.moves, self.list_encode_opponent_team_move, self.size_opponent_team_move))
        for i in range(0, 6-len(battle.opponent_team)):
            encoded_return.append(np.zeros(self.size_opponent_team))
            encoded_return.append(np.zeros(4*self.size_opponent_team_move))
        
        return np.concatenate(encoded_return)

    def _encode_battle(self, battle: AbstractBattle) -> np.ndarray:
        encoded_return = []
        for encode in self.list_encode_battle:
            encoded_return.append(encode(battle))
        return np.concatenate(encoded_return)

    def _encode_pokemon(self, pokemon: Pokemon, list_encode: List[Callable]) -> np.ndarray:
        encoded_return = []
        for encode in list_encode:
            encoded_return.append(encode(pokemon))
        return np.concatenate(encoded_return)

    def _encode_moves(self, moves: Dict[str, Move], list_encode: List[Callable], size_move: int) -> np.ndarray:
        encoded_return = []
        for _, move in moves.items():
            for encode in list_encode:
                encoded_return.append(encode(move))
        for i in range(0, 4-len(moves)):
            encoded_return.append(np.zeros(size_move))
        return np.concatenate(encoded_return)
