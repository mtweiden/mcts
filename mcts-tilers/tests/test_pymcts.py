"""
In-process MCTS over a real tilers env with a Python agent.

`PyMcts.run(env, agent, ...)` takes a tilers `Environment` and an
`MctsAgent` wrapping a Python object exposing
``infer_from_obs(obs_list) -> (priors, values)``.  The observation dicts
come from the Rust adapter (`obs_to_pydict`) and carry the 10-channel
`board` plus `valid_actions` (encoded `u16` ids).

`mcts_tilers` statically links its own copy of `tilers`, so the env must
be `mcts_tilers.Environment` (its embedded type) — an `Environment` from
the standalone `tilers` package is a *different* Python class and PyMcts
would reject it.

The real `tile.Agent.infer_from_obs` is intentionally stubbed, so this
in-process path uses a trivial uniform agent; the real model is wired
through the IPC handler instead (see `test_board_ipc.py`).
"""

from typing import Any

from mcts_tilers import Environment, MctsAgent, PyMcts


class DummyAgent:
    """Uniform prior over each obs's valid actions; value 0."""

    def infer_from_obs(
        self, obs_list: list[dict[str, Any]]
    ) -> tuple[list[dict[int, float]], list[float]]:
        priors = []
        for o in obs_list:
            va = o["valid_actions"]
            priors.append({a: 1.0 / len(va) for a in va} if va else {})
        values = [0.0 for _ in obs_list]
        return priors, values


def make_env() -> Environment:
    env = Environment(4, 4, 3)
    env.set_seed(0)
    env.random_start(2, non_fault_tolerant_mode=True)
    return env


class TestPyMcts:
    def test_init_pymcts(self) -> None:
        assert isinstance(PyMcts(batch_size=4), PyMcts)

    def test_agent(self) -> None:
        assert isinstance(MctsAgent(DummyAgent()), MctsAgent)

    def test_run_mcts(self) -> None:
        env = make_env()
        node = PyMcts(batch_size=1).run(env, MctsAgent(DummyAgent()), num_steps=64)
        assert isinstance(node.id(), int)
        # The root expanded its children → it accrued edge visits.
        assert sum(node.edge_visits().values()) > 0

    def test_run_observation_carries_board(self) -> None:
        # The agent is handed obs dicts from `obs_to_pydict`: 10-channel
        # board + encoded valid-action ids.
        captured = {}

        class CapturingAgent(DummyAgent):
            def infer_from_obs(self, obs_list):
                captured["obs"] = obs_list[0]
                return super().infer_from_obs(obs_list)

        env = make_env()
        PyMcts(batch_size=1).run(env, MctsAgent(CapturingAgent()), num_steps=8)
        obs = captured["obs"]
        assert "board" in obs and "valid_actions" in obs
        assert obs["num_layers"] == 2  # lookahead (1) + 1
        # board: [layer][cell][10]
        assert len(obs["board"][0][0]) == 10

    def test_advance_root(self) -> None:
        env = make_env()
        mcts = PyMcts(batch_size=1)
        agent = MctsAgent(DummyAgent())

        node = mcts.run(env, agent, num_steps=64)
        visits = node.edge_visits()
        assert visits, "root should have visited children"
        best = max(visits, key=visits.get)

        mcts.advance_root(best)
        # Re-running from the advanced root still produces a valid search.
        node2 = mcts.run(env, agent, num_steps=32)
        assert sum(node2.edge_visits().values()) >= 0

    def test_full_game_progresses(self) -> None:
        # Drive a short game entirely from Python: search, take the most-
        # visited action, decode the u16 id back to an Action via
        # `mcts_tilers.rl`, step the env, repeat.  All types come from
        # `mcts_tilers` so they interop (no cross-module mismatch).
        import mcts_tilers

        env = make_env()
        agent = MctsAgent(DummyAgent())
        mcts = PyMcts(batch_size=1)

        steps = 0
        while not env.done() and steps < 20:
            node = mcts.run(env, agent, num_steps=48)
            visits = node.edge_visits()
            if not visits:
                break
            best_id = int(max(visits, key=visits.get))
            action = mcts_tilers.rl.decode(env, best_id)
            env.step(action)
            mcts.advance_root(best_id)
            steps += 1

        assert steps > 0, "the game should take at least one action"
