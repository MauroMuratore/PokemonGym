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

def test_dqn_agent():
    battle_format ="gen4anythinggoes"
    configuration = AccountConfiguration("Trainer", None)
    opponent_configuration = AccountConfiguration("Opponent", None)

    team_support = TeamSupport("data/teams/current")
    opponent_team_support = TeamSupport("data/teams/current")
    opponent_player = SimpleHeuristicsPlayer(
        battle_format=battle_format,
        account_configuration=opponent_configuration,
        team=opponent_team_support
    )

    battle_env = BattleEnv(
        battle_format=battle_format,
        account_configuration1=configuration,
        log_level=25,
        strict=False,
        team=team_support
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
    target_network = ForwardNN(size_observation, size_action).to(device)
    target_network.load_state_dict(policy_network.state_dict())

    master = SimpleHeuristicsPlayer()

    optimizer = optim.Adam(policy_network.parameters(), lr=5e-4, amsgrad=True)
    memory = ReplayMemory(capacity = 10000)

    dqn_agent = DQNAgent(policy_network, 
                        target_network, 
                        env=wrapper_env, 
                        memory=memory,
                        optimizer=optimizer,
                        device=device,
                        master=master,
                        )

    tic = time.time()
    total_episode = 50
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
        wrapper_env.close()
        print(f"training lasts {days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}")

if __name__ == "__main__":
    test_dqn_agent()