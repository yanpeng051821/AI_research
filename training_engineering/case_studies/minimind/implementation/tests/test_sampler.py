import pytest
import torch

from minimind_lab.data.sampler import StatefulSampler


def test_sampler_restores_committed_not_fetched_position():
    rng = torch.get_rng_state().clone()
    sampler = StatefulSampler(5, seed=42)
    assert sampler.indices == [2, 4, 3, 0, 1]
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    iterator = iter(sampler)
    assert [next(iterator) for _ in range(4)] == [2, 4, 3, 0]
    sampler.commit(2)
    state = sampler.state_dict()
    state['indices'][0] = 99
    assert sampler.indices[0] == 2
    state = sampler.state_dict()
    restored = StatefulSampler(5, seed=99)
    restored.load_state_dict(state)
    state['indices'][0] = 99
    assert restored.indices[0] == 2
    assert restored.position == restored.committed_position == 2
    assert list(restored) == [3, 0, 1]
    assert list(restored) == []


@pytest.mark.parametrize('state', [
    {'indices': [0, 1], 'committed_position': 0},
    {'indices': [0, 0, 2], 'committed_position': 0},
    {'indices': [0, 1, 3], 'committed_position': 0},
    {'indices': [0, 1, 2.0], 'committed_position': 0},
    {'indices': [False, 1, 2], 'committed_position': 0},
    {'indices': [0, 1, 2], 'committed_position': -1},
    {'indices': [0, 1, 2], 'committed_position': 4},
    {'indices': [0, 1, 2], 'committed_position': True},
])
def test_sampler_rejects_invalid_state_without_mutation(state):
    sampler = StatefulSampler(3)
    next(iter(sampler))
    before = sampler.state_dict()
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)
    assert sampler.state_dict() == before
    assert sampler.position == 1


def test_sampler_commit_boundaries_and_empty_input():
    sampler = StatefulSampler(3)
    next(iter(sampler))
    for count in (-1, 2):
        with pytest.raises(ValueError):
            sampler.commit(count)
        assert sampler.committed_position == 0
    sampler.commit(0)
    sampler.commit(1)
    assert sampler.committed_position == 1
    empty = StatefulSampler(0)
    empty.load_state_dict(empty.state_dict())
    assert list(empty) == []
