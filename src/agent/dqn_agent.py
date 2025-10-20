import math
import random
import torch
import torch.nn as nn
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.player import Player
from poke_env.player.baselines import RandomPlayer
from src.memory.replay_memory import ReplayMemory
from src.memory.transition import Transition 
from src.environment.battle_env import BattleEnv
from itertools import count



class DQNAgent():

    def __init__(self, policy_net: nn.Module, 
                target_net: nn.Module,
                env : BattleEnv,
                memory : ReplayMemory,
                optimizer,
                batch_size :int  = 64,
                gamma :float = 0.95,
                eps_start :float = 0.9 ,
                eps_end :float = 0.01,
                eps_decay : int = 2500,
                tau :float=0.005,
                device :str ="cpu",
                master : Player = None
                ):
        
        self.policy_net = policy_net
        self.target_net = target_net
        self.env = env 
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start =eps_start
        self.eps_end =eps_end
        self.eps_decay =eps_decay
        self.tau = tau
        self.optimizer = optimizer
        self.device = device
        if master:
            self.master = master
        else:
            self.master = RandomPlayer()
        self.n_episode = 0
    
    def act(self, state) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1,1)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def act_training(self, state, step):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step / self.eps_decay)
        actor = 0
        if sample > eps_threshold:
            action = self.act(state)
            actor = 1
        else:
            battle = self.env.current_battle()
            if battle:
                order = self.master.choose_move(battle)
                action = SinglesEnv.order_to_action(order, battle, strict=False)
                if action < 0:
                    action = 10 + action
                action = torch.tensor([[action]], device=self.device, dtype=torch.long)
            else:
                action = random.randint(0,self.env.action_space.n -1)
                action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        return action, actor
    
    def train(self, n_episodes = 1000):
        step = 0
        print("start_training") 
        for episode in range(n_episodes):
            self.n_episode +=1 
            action_master = 0
            action_policy = 0
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for n_step in count():
                step += 1
                action, actor = self.act_training(state, step)
                if actor == 1:
                    action_policy +=1
                else:
                    action_master +=1
                observation, reward, terminated, truncated, _ = self.env.step(action)
                reward = torch.tensor([reward], device = self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                self.memory.push(state, action, next_state, reward)

                state = next_state

                loss = self.optimize_model()
                if not loss:
                    loss = -1

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key]=self.tau * policy_net_state_dict[key] + (1-self.tau)*target_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    print(f"episode: {episode} step: {n_step} action ({action_master},{action_policy}) loss: {loss:.5f} reward: {reward.item():.5f} memory size: {len(self.memory)}")
                    break
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)


    def set_master(self, master: Player):
        self.master = master
