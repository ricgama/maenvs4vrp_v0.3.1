"""Instance generators: input validation, offline use and toy instances."""
import importlib

import pytest

from tests.helpers import AEC_ENVS, PARALLEL_ENVS, PKG_DIR, make_env, module, nodes_for, rollout

GENERATORS = [("environments", e) for e in AEC_ENVS] + [("parallel_environments", e) for e in PARALLEL_ENVS]
IDS = [f"{g[:3]}-{e}" for g, e in GENERATORS]
TOY_GENERATORS = [(g, e) for g, e in GENERATORS if (PKG_DIR / g / e / "toy_instance_generator.py").exists()]


@pytest.mark.parametrize("group, env_name", GENERATORS, ids=IDS)
def test_unknown_sample_type_raises(group, env_name):
    gen = module(env_name, "instances_generator", group).InstanceGenerator()
    with pytest.raises(ValueError, match="sample_type"):
        gen.sample_instance(num_agents=2, num_nodes=nodes_for(env_name, 11), sample_type="unknown")


@pytest.fixture
def no_network(monkeypatch):
    """Make every Hugging Face call fail, as if there was no network access."""
    def offline(*args, **kwargs):
        raise ConnectionError("network access is not allowed in this test")

    import huggingface_hub
    for name in ("snapshot_download", "hf_hub_download"):
        monkeypatch.setattr(huggingface_hub, name, offline, raising=False)
    for method in ("repo_exists", "list_repo_files", "repo_info", "dataset_info"):
        monkeypatch.setattr(huggingface_hub.HfApi, method, offline, raising=False)
    for group, env_name in GENERATORS:
        mod = module(env_name, "instances_generator", group)
        if hasattr(mod, "snapshot_download"):
            monkeypatch.setattr(mod, "snapshot_download", offline)


@pytest.mark.parametrize("group, env_name", GENERATORS, ids=IDS)
def test_random_generation_works_offline(group, env_name, no_network):
    gen = module(env_name, "instances_generator", group).InstanceGenerator()
    instance = gen.sample_instance(num_agents=2, num_nodes=nodes_for(env_name, 11))
    assert instance["data"]["coords"].shape[-1] == 2
    augmented = gen.sample_instance(num_agents=2, num_nodes=nodes_for(env_name, 11), batch_size=4,
                                    sample_type="augment", n_augment=2)
    assert augmented["data"].batch_size[0] == 4


@pytest.mark.parametrize("group, env_name", TOY_GENERATORS, ids=[f"{g[:3]}-{e}" for g, e in TOY_GENERATORS])
def test_toy_instance_episode(group, env_name):
    toy = module(env_name, "toy_instance_generator", group).ToyInstanceGenerator()
    env = make_env(env_name, group=group, generator=toy, batch_size=None)
    td = rollout(env, "select" if group == "environments" else "parallel")
    assert td["done"].all()
    env.check_solution_validity()


def test_pdptw_requires_odd_number_of_nodes():
    gen = module("pdptw", "instances_generator").InstanceGenerator()
    with pytest.raises(ValueError, match="odd"):
        gen.sample_instance(num_agents=2, num_nodes=10)
