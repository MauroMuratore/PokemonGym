from src.utils.team_support import TeamSupport
from poke_env.player.baselines import SimpleHeuristicsPlayer, RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration

class FactoryPlayer():
    def __init__(self):
        super().__init__()

    def create_player(self, config):
        assert "type_trainer" in config 
        assert "battle_format" in config
        assert "configuration" in config

        if config["type_trainer"] == "simple_heuristic":
            if "team_support" in config:
                return SimpleHeuristicsPlayer(
                        battle_format = config["battle_format"],
                        account_configuration = AccountConfiguration(
                            config["configuration"]["username"],
                            config["configuration"]["password"]
                            ),
                        team = TeamSupport(config["team_support"])
                        )
            else:
                SimpleHeuristicsPlayer(
                        battle_format = config["battle_format"],
                        account_configuration = AccountConfiguration(
                            config["configuration"]["username"],
                            config["configuration"]["password"]
                            )
                        )
        else:
            if "team_support" in config:
                return RandomPlayer(
                        battle_format = config["battle_format"],
                        account_configuration = AccountConfiguration(
                            config["configuration"]["username"],
                            config["configuration"]["password"]
                            ),
                        team = TeamSupport(config["team_support"])
                        )
            else:
                RandomPlayer(
                        battle_format = config["battle_format"],
                        account_configuration = AccountConfiguration(
                            config["configuration"]["username"],
                            config["configuration"]["password"]
                            )
                        )

