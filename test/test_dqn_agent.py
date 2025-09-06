import os
import re
import time
import torch
import torch.optim as optim
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer, MaxBasePowerPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from src.network.forward_nn import ForwardNN
from src.agent.dqn_agent import DQNAgent
from src.environment.battle_agent_wrapper import BattleAgentWrapper
from src.environment.battle_env import BattleEnv
from src.memory.replay_memory import ReplayMemory
from src.utils.team_support import TeamSupport

def get_next_dqn_filename(folder_path: str, prefix: str = "dqn", ext: str = "pt") -> str:
    """
    Restituisce il primo nome disponibile con struttura dqn_n all'interno della cartella.
    :param folder_path: percorso della cartella da controllare
    :param prefix: prefisso del file (default: "dqn")
    :return: stringa col nome file disponibile, es: "dqn_3"
    """
    pattern = re.compile(rf"^{prefix}_(\d+)\.{ext}$")
    used_numbers = set()

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            used_numbers.add(int(match.group(1)))

    n = 1
    while n in used_numbers:
        n += 1

    return f"{prefix}_{n}.{ext}"

def test_dqn_agent():
    #battle_format = "gen4anythinggoes"
    battle_format ="gen4randombattle"
    team = TeamSupport("data/teams/current")
    opponent_team = TeamSupport("data/teams/current")
    configuration = AccountConfiguration("Trainer", None)
    opponent_configuration = AccountConfiguration("Opponent", None)

    opponent_player = SimpleHeuristicsPlayer(
        battle_format=battle_format,
        account_configuration=opponent_configuration,
        #team=opponent_team
    )

    battle_env = BattleEnv(
        battle_format=battle_format,
        account_configuration1=configuration,
        #team=team,
        log_level=25,
        strict=False
    )

    wrapper_env = BattleAgentWrapper(
        battle_env,
        opponent_player
    )

    size_observation =wrapper_env.observation_space.shape[0]
    size_action = wrapper_env.action_space.n

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    policy_network = ForwardNN(size_observation, size_action).to(device)
    #policy_network.load_state_dict(torch.load("model/dqn_1.pt", weights_only=True))
    target_network = ForwardNN(size_observation, size_action).to(device)
    target_network.load_state_dict(policy_network.state_dict())

    master = SimpleHeuristicsPlayer()

    optimizer = optim.Adam(policy_network.parameters(), lr=5e-4, amsgrad=True)
    memory = ReplayMemory(10000)

    dqn_agent = DQNAgent(policy_network, 
                        target_network, 
                        env=wrapper_env, 
                        memory=memory,
                        optimizer=optimizer,
                        device=device,
                        master=master,
                        )

    model_name = "model/" + get_next_dqn_filename("model")
    tic = time.time()
    total_episode = 20_000
    try:
        dqn_agent.train(total_episode)
    finally:
        toc = time.time()
        seconds = int(toc - tic)
        minutes = seconds // 60
        seconds = seconds % 60
        hours = minutes // 60
        minutes = minutes % 60
        days = hours // 24
        hours = hours % 24
        dqn_agent.save_model(model_name)
        wrapper_env.close()
        print(f"training lasts {days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}")

if __name__ == "__main__":
    test_dqn_agent()