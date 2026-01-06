from typing import Any
from pymcts import MctsAgent
from pymcts import MctsEnvironment
from pymcts import PyMcts

class DummyAgent:
    def infer(self, obs: list[dict[str, Any]]) -> tuple[list[dict[int, float]], list[float]]:
        prior_probs = [
            {a: 1.0 / len(o['valid_actions']) for a in o['valid_actions']}
            for o in obs
        ]
        values = [0.0 for _ in range(len(obs))]
        return prior_probs, values

class DummyEnvironment:
    """
    Try to increment a counter until a value is reached.
    """
    def __init__(self, target_value: int) -> None:
        self.target_value = target_value
        self.current_value = 0

    def step(self, action: int) -> None:
        assert action in [0, 1]
        if action == 0:
            self.current_value += 1
        else:
            if self.current_value > 0:
                self.current_value -= 1

    def done(self) -> bool:
        return self.current_value >= self.target_value

    def observation(self) -> dict[str, Any]:
        if self.current_value > 0:
            valid_actions = [0, 1]
        else:
            valid_actions = [0]
        return {
            'placement': [self.current_value],
            'objectives_0': [],
            'objectives_1': [],
            'height': 1,
            'width': 1,
            'num_ancillas': 0,
            'valid_actions': valid_actions,
        }

    def valid_actions(self) -> list[int]:
        return [0, 1]

    def hash_state(self) -> int:
        return hash(self.current_value)

    def render(self) -> str:
        return f"{self.current_value} ({self.target_value})"

class TestPyMcts:
    def test_init_pymcts(self) -> None:
        mcts = PyMcts(terminal_value=0.5, batch_size=4)
        assert isinstance(mcts, PyMcts)
    
    def test_agent(self) -> None:
        dummy_agent = DummyAgent()
        mcts_agent = MctsAgent(dummy_agent)
        assert isinstance(mcts_agent, MctsAgent)
    
    def test_environment(self) -> None:
        dummy_env = DummyEnvironment(target_value=5)
        mcts_env = MctsEnvironment(dummy_env)
        assert isinstance(mcts_env, MctsEnvironment)
    
    def test_run_mcts(self) -> None:
        dummy_agent = DummyAgent()
        mcts_agent = MctsAgent(dummy_agent)
        dummy_env = DummyEnvironment(target_value=5)
        mcts_env = MctsEnvironment(dummy_env)
        mcts = PyMcts(terminal_value=1.0, batch_size=1)
        mcts_node = mcts.run(mcts_env, mcts_agent, num_steps=100)
        assert mcts_node.id() == 0
        # assert hasattr(mcts_node, 'id')
        # assert hasattr(mcts_node, 'prior_probs')
        # assert hasattr(mcts_node, 'value')
        # assert hasattr(mcts_node, 'terminal_state')
        # assert hasattr(mcts_node, 'repr')
    
    def test_full_loop(self) -> None:
        agent = DummyAgent()
        env = DummyEnvironment(target_value=5)
        mcts_agent = MctsAgent(agent)
        mcts_env = MctsEnvironment(env)
        mcts = PyMcts(terminal_value=1.0, batch_size=1)

        while not env.done():
            node = mcts.run(mcts_env, mcts_agent, num_steps=100)
            best_visits = -1
            for action, visits in node.edge_visits().items():
                if visits > best_visits:
                    best_visits = visits
                    best_action = action
            env.step(best_action)
        assert env.current_value >= env.target_value