from gymnasium.utils.env_checker import check_env
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer, RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from src.utils.team_support import TeamSupport
from src.environment.battle_env import BattleEnv

from src.utils import encode_battle
import asyncio

def test_battle_env():
    battle_format = "gen4anythinggoes"
    team = TeamSupport("data/teams/current")
    opponent_team = TeamSupport("data/teams/current", count=1)
    account_configuration = AccountConfiguration("Trainer", None)
    opponent_account_configuration = AccountConfiguration("Opponent", None)

    simple_heuristic_player = SimpleHeuristicsPlayer(
        battle_format= battle_format,
        team=opponent_team,
        account_configuration=opponent_account_configuration        
    )

    battle_env = BattleEnv(
        battle_format=battle_format,
        account_configuration1=account_configuration,
        #server_configuration=LocalhostServerConfiguration,
        team=team,
        log_level=25,
        strict=False,
    )

    wrapper_env = SingleAgentWrapper(
        battle_env,
        simple_heuristic_player
    )
    try:
        check_env(wrapper_env)
    finally:
        wrapper_env.close()

async def test_battle():
    battle_format = "gen4anythinggoes"
    team = TeamSupport("data/teams/current")
    opponent_team = TeamSupport("data/teams/current", count=1)
    account_configuration = AccountConfiguration("Trainer", None)
    opponent_account_configuration = AccountConfiguration("Opponent", None)
    
    player = SimpleHeuristicsPlayer(
        battle_format= battle_format,
        team=team,
        account_configuration=account_configuration        
    )

    opponent_player = SimpleHeuristicsPlayer(
        battle_format= battle_format,
        team=opponent_team,
        account_configuration=opponent_account_configuration        
    )

    await player.battle_against(opponent_player)    


if __name__ == "__main__":
    test_battle_env()
    #asyncio.run(test_battle())