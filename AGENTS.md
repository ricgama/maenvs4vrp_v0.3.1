# AGENTS.md - MAEnvs4VRP (Multi-Agent Environments for Vehicle Routing Problems)

PyTorch/TensorDict library of multi-agent environments for vehicle routing problems,
following an Agent-Environment-Cycle (AEC) design inspired by PettingZoo.
Paper: Gama et al., *INFORMS Journal on Computing* (2026), doi:10.1287/ijoc.2025.1211.

## Repository layout
- `maenvs4vrp/core/` - abstract base classes: `AECEnv` (`env.py`), `PEnv` (`parallel_env.py`),
  `InstanceBuilder`, `ObservationBuilder`, `BaseSelector`, `RewardFn`.
- `maenvs4vrp/environments/<problem>/` - one self-contained package per problem (AEC, one agent acts per step):
  `cvrp cvrptw cvrpstw dvrptw dsvrptw hcvrp mdvrptw pcvrp pcvrptw pdptw sdvrptw top toptw mtvrp mtdvrp gmtvrp gmtdvrp`.
- `maenvs4vrp/parallel_environments/<problem>/` - parallel variants (all agents act per step): `cvrp pcvrp pcvrptw top toptw`.
- `maenvs4vrp/neuro_solvers/` - reference neural solvers (attention model, 2D pointer) and training scripts.
- `maenvs4vrp/utils/` - shared helpers (`utils.py`, `ops.py`, `plotting.py`, `augment_utils.py`).
- `maenvs4vrp/learning_notebooks/` - tutorial notebooks (source of truth) and exercise `snippets/`.
- `docs/source/` - Sphinx docs (furo theme, napoleon, nbsphinx). One folder per environment mirroring the code.
- `tests/unit/` - pytest suite.

## Core design rule: isolated but uniform environments
Each environment folder is **deliberately self-contained**: code is duplicated between environments
instead of being shared, so a problem can be read, copied and modified in isolation. Do **not**
refactor duplicated code into shared helpers. Instead, keep every folder structurally identical:

| File | Required content |
|---|---|
| `__init__.py` | present (may be empty) |
| `env.py` | `Environment(AECEnv)` (or `PEnv` for parallel envs) |
| `instances_generator.py` | `InstanceGenerator(InstanceBuilder)` - random instances |
| `toy_instance_generator.py` | `ToyInstanceGenerator(InstanceBuilder)` - small hand-made instances |
| `benchmark_instances_generator.py` | `BenchmarkInstanceGenerator(InstanceBuilder)` - only when public benchmarks exist |
| `observations.py` | `Observations(ObservationBuilder)` with the `POSSIBLE_*_FEATURES` lists |
| `env_agent_selector.py` | `AgentSelector`, `RandomSelector`, `SmallestTimeAgentSelector` |
| `env_agent_reward.py` | `DenseReward`, `SparseReward` |

- Known gaps: `top` (both groups) has no `toy_instance_generator.py`; `parallel_environments/cvrp` has no
  `env_agent_selector.py`; `dvrptw`, `dsvrptw`, `cvrp` and `pcvrp` have no benchmark generator.
- Public method names and signatures (`reset`, `observe`, `step`, `sample_action`, `check_solution_validity`, ...)
  must stay identical across environments; problem-specific parameters are added as keyword arguments.
- State keys in `td_state` should use the same names in every environment (e.g. `demands`, `coords`, `tw_low`, `tw_high`).
  Known legacy exception: `hcvrp` uses `demand` / `capacity`.
- When fixing a bug or changing an API in one environment, apply the same change to **every** environment
  that has the same code (grep for it) and to the matching `docs/source/<group>/<problem>/` pages.
- Generators must raise `ValueError` for an unknown `sample_type`, and random generation must work offline.

## Environment & commands
- Python >= 3.11. Development conda env: `maenvs4vrp`
  (`/home/gama/miniconda3/envs/maenvs4vrp/bin/python`). It has an editable install of a *different*
  checkout (`maenvs4vrp_dev`), so run with `PYTHONPATH=$PWD` (pytest already does via `pytest.ini`).
- Install: `pip install -e ".[dev,docs]"` (or `uv sync --all-extras`).
- Tests: `pytest tests/` (slow - covers every environment). Run the files relevant to your change, e.g.
  `pytest tests/unit/environments/reset_test.py -k cvrptw`.
- Smaller solution checks: `pytest tests/unit/environments/check_solution_test.py --nodes 21 --agents 3`.
- Docs: `cd docs && make html` (output in `docs/build/`, not committed). Read the Docs builds with
  `fail_on_warning: true`, so check locally with `python -m sphinx -W --keep-going -b html docs/source /tmp/docs`.
- Observation docs pages are generated from the code: `python docs/tools/generate_observations_docs.py`
  (it also reports `POSSIBLE_*_FEATURES` entries without a `get_feat_*` method).
- Notebooks: `jupyter nbconvert --to notebook --execute --output-dir /tmp/nb <nb>.ipynb`, run from
  `maenvs4vrp/learning_notebooks/`. They are the single source; `docs/source/notebooks/` is filled at docs build time.
- Bibliography: cite with `[Key]_` and add entries only to `docs/source/content/references.rst`.

## Benchmark data
Benchmark instances are downloaded on demand from the Hugging Face dataset `MAL4VRP/maenvs4vrp`
into `maenvs4vrp/environments/<problem>/data/`. These folders are git-ignored; never commit data,
`docs/build/`, `dist/`, `*.egg-info` or `__pycache__`.

## Code style
- Match the surrounding code: 4-space indentation, snake_case functions, PascalCase classes,
  imports ordered stdlib / third-party / local.
- Tensors: batch dimension first; document shapes as `[B, N]`, `[B, A, N]` (B=batch, A=agents, N=nodes).
- Use type annotations in signatures of public functions.

## Docstrings (Google style, enforced across all environments)
```python
def reset(self, num_agents: int | None = None, seed: int | None = None) -> TensorDict:
    """
    Reset the environment and load a new batch of instances.

    Args:
        num_agents (int, optional): Total number of agents. Defaults to None.
        seed (int, optional): Random number generator seed. Defaults to None.

    Returns:
        TensorDict: Environment state after reset.

    Raises:
        ValueError: If ``sample_type`` is unknown.
    """
```
- Every module, public class and public method has a docstring; module docstrings are one line describing the file.
- Section headers end with `:` (`Args:`, `Returns:`, `Raises:`). Parameters: `name (type): Description.`
  with a space before the parenthesis; optional parameters add `, optional` and end with `Defaults to X.`
- List **exactly** the parameters in the signature (no removed parameters, no missing ones).
- Omit `Args:` when there are no parameters and omit `Returns:` when nothing is returned
  (no `N/a`, no `Returns: None.`). `__init__` never has a `Returns:` section.
- The same method must have the same docstring wording in every environment, except for problem-specific details.

## Contributions
- Keep diffs minimal and focused; no unrelated reformatting.
- Preserve backwards compatibility of public APIs unless fixing a real bug; update notebooks and docs when an API changes.
- Commit messages: short imperative summary line.
