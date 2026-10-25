import copy

import pytest
import torch
from torch.utils.data import DataLoader

from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training import checkpoint, loop
from minimind_lab.training.step import train_update


def make_objects(seed=42):
    torch.manual_seed(seed)
    model = MinimindForCausalLM(
        ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler


def make_loader():
    rows = torch.tensor([[1, token, 2] for token in range(3, 8)])
    dataset = [dict(input_ids=row, attention_mask=torch.ones_like(row), labels=row.clone()) for row in rows]
    sampler = StatefulSampler(5, seed=42)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler,
                        generator=torch.Generator().manual_seed(123))
    return loader, sampler


def assert_tree_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_tree_close(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-7)
    else:
        assert actual == expected


def test_interrupted_training_matches_continuous(tmp_path):
    a, oa, sa = make_objects()
    la, ia = make_loader()
    history_a = loop.train_epoch(a, oa, la, sa, sampler=ia)
    b, ob, sb = make_objects()
    lb, ib = make_loader()
    iterator = iter(lb)
    window = [next(iterator), next(iterator)]
    first = train_update(b, ob, window, sb)
    ib.commit(sum(batch['input_ids'].shape[0] for batch in window))
    path = checkpoint.save_checkpoint(tmp_path/'state.pt', b, ob, sb, 1, ib)
    c, oc, sc = make_objects(seed=999)
    lc, ic = make_loader()
    step = checkpoint.load_checkpoint(path, c, oc, sc, ic)
    assert step == 1 and ic.position == ic.committed_position == 4
    assert ic.indices[ic.position:] == [1]
    history_c = loop.train_epoch(c, oc, lc, sc, sampler=ic)
    assert_tree_close([first] + history_c, history_a)
    assert_tree_close(c.state_dict(), a.state_dict())
    assert_tree_close(oc.state_dict(), oa.state_dict())
    assert sc.state_dict() == sa.state_dict()
    assert ic.state_dict() == ia.state_dict()
    assert step + len(history_c) == 2


def test_loop_commits_only_successful_windows(monkeypatch):
    model, optimizer, scheduler = make_objects()
    loader, sampler = make_loader()
    calls = 0

    def fail_on_tail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('tail failed')
        return train_update(*args, **kwargs)

    monkeypatch.setattr(loop, 'train_update', fail_on_tail)
    with pytest.raises(RuntimeError, match='tail failed'):
        loop.train_epoch(model, optimizer, loader, scheduler, sampler=sampler)
    assert sampler.position == 5
    assert sampler.committed_position == 4
    assert scheduler.last_epoch == 1


def test_checkpoint_restores_random_streams(tmp_path):
    model, optimizer, scheduler = make_objects()
    path = checkpoint.save_checkpoint(tmp_path/'state.pt', model, optimizer, scheduler, 0)
    expected_cpu = torch.rand(5)
    expected_cuda = torch.rand(5, device='cuda') if torch.cuda.is_available() else None
    torch.rand(17)
    if expected_cuda is not None:
        torch.rand(19, device='cuda')
    restored = make_objects(seed=999)
    assert checkpoint.load_checkpoint(path, *restored) == 0
    torch.testing.assert_close(torch.rand(5), expected_cpu, rtol=0, atol=0)
    if expected_cuda is not None:
        torch.testing.assert_close(torch.rand(5, device='cuda'), expected_cuda, rtol=0, atol=0)


@pytest.mark.parametrize('missing_key', [False, True])
def test_missing_sampler_state_fails_before_model_load(tmp_path, missing_key):
    model, optimizer, scheduler = make_objects()
    path = checkpoint.save_checkpoint(tmp_path/'state.pt', model, optimizer, scheduler, 0)
    if missing_key:
        state = torch.load(path, weights_only=True)
        del state['sampler_state']
        torch.save(state, path)
    restored = make_objects(seed=999)
    before = copy.deepcopy(restored[0].state_dict())
    with pytest.raises(ValueError, match='no sampler state'):
        checkpoint.load_checkpoint(path, *restored, sampler=StatefulSampler(5))
    assert_tree_close(restored[0].state_dict(), before)


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    objects = make_objects()
    path = checkpoint.save_checkpoint(tmp_path/'state.pt', *objects, 0)
    before = path.read_bytes()

    def fail_save(state, target):
        target.write(b'incomplete')
        raise OSError('simulated write failure')

    monkeypatch.setattr(checkpoint.torch, 'save', fail_save)
    with pytest.raises(OSError, match='simulated write failure'):
        checkpoint.save_checkpoint(path, *objects, 1)
    assert path.read_bytes() == before
    assert not path.with_name(path.name + '.tmp').exists()
