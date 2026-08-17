"""Tests for the BO loop's compute settings.

The default of one torch thread is a REPRODUCIBILITY requirement, not a
performance choice: above one thread, BLAS reorders its reductions, the GP fit
moves in its last bits, and the search — a feedback loop — amplifies that to
orders of magnitude within ~13 iterations. See `utils.DEFAULT_BO_THREADS`.

The cap is also a global, and a library that silently rewires a caller's torch
is a bug, so the restore behaviour is pinned here too.
"""

import pytest
import torch

from bayesian_pid.utils import DEFAULT_BO_THREADS, torch_thread_limit


def test_thread_limit_applies_inside_the_block() -> None:
    with torch_thread_limit(2):
        assert torch.get_num_threads() == 2


def test_thread_limit_restores_the_previous_setting() -> None:
    before = torch.get_num_threads()
    with torch_thread_limit(1):
        pass
    assert torch.get_num_threads() == before


def test_thread_limit_restores_even_when_the_block_raises() -> None:
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="boom"), torch_thread_limit(1):
        raise RuntimeError("boom")
    assert torch.get_num_threads() == before


def test_none_leaves_the_setting_alone() -> None:
    before = torch.get_num_threads()
    with torch_thread_limit(None):
        assert torch.get_num_threads() == before
    assert torch.get_num_threads() == before


def test_zero_threads_is_rejected() -> None:
    before = torch.get_num_threads()
    with pytest.raises(ValueError, match="at least 1"):
        with torch_thread_limit(0):
            pass
    assert torch.get_num_threads() == before


def test_default_is_single_threaded() -> None:
    """Load-bearing. A seeded run only replays exactly at one thread, so this
    default is a correctness property. Raising it silently would make every
    recorded result unreproducible without anything failing.
    """
    assert DEFAULT_BO_THREADS == 1
