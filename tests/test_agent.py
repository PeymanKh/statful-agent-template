from src.agents.agent import Agent

def test_agent_run():
    agent = Agent()
    assert agent.run("test") == "test"
