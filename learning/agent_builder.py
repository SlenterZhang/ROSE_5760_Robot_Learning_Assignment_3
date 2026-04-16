import yaml

import scripts.ppo_agent as ppo_agent
import scripts.sac_agent as sac_agent
import scripts.dyna_agent as dyna_agent
import scripts.bc_dagger_agent as bc_agent 

def build_agent(agent_file, env, device):
    agent_config = load_agent_file(agent_file)
    
    agent_name = agent_config["agent_name"]
    print("Building {} agent".format(agent_name))

    if (agent_name == ppo_agent.PPOAgent.NAME):
        agent = ppo_agent.PPOAgent(config=agent_config, env=env, device=device)
    elif (agent_name == sac_agent.SACAgent.NAME):
        agent = sac_agent.SACAgent(config=agent_config, env=env, device=device)
    elif (agent_name == dyna_agent.DYNAAgent.NAME):
        agent = dyna_agent.DYNAAgent(config=agent_config, env=env, device=device)
    elif (agent_name == bc_agent.BCDAggerAgent.NAME):
        agent = bc_agent.BCDAggerAgent(config=agent_config, env=env, device=device)        
    else:
        assert(False), "Unsupported agent: {}".format(agent_name)

    return agent

def load_agent_file(file):
    with open(file, "r") as stream:
        agent_config = yaml.safe_load(stream)
    return agent_config
