# An implementation of MCTS that uses remote inference
A server running an inference agent can be launched by running the `server/server.py` script.

Running `cargo run --bin gather --release` will launch Gather processes that run MCTS and query the server for prior weights.

A `dummy` version of the server can also be run with `cargo run --bin server --release`.