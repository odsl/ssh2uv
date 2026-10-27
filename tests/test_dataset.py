import numpy as np
import pytest

from ssh2uv.data import create_stacked_train_val_datasets

CHANNELS = ["Eta", "windx", "windy", "theta", "u1", "v1"]


@pytest.fixture
def stacked_dir(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(60, len(CHANNELS), 64, 64)).astype(np.float32)
    np.save(tmp_path / "global-final-64_stacked_DEEP.npy", data)
    return tmp_path, data


def make(path, seed=42):
    return create_stacked_train_val_datasets(
        data_dir=str(path), data_file="global-final-64_stacked_DEEP.npy", channel_order=CHANNELS,
        input_vars=["Eta", "theta"], output_vars=["u1", "v1"], val_fraction=0.2, normalize=False,
        seed=seed, use_precomputed_split=True, verbose=False)


def test_random_split_is_deterministic_and_disjoint(stacked_dir):
    path, _ = stacked_dir
    tr1, va1 = make(path)
    tr2, va2 = make(path)
    assert np.array_equal(va1.indices, va2.indices)
    assert len(va1) == 12 and len(tr1) == 48
    assert not set(tr1.indices) & set(va1.indices)


def test_sample_channels(stacked_dir):
    path, data = stacked_dir
    _, va = make(path)
    x, y, mask = va[0]
    k = va.indices[0]
    assert np.allclose(x.numpy(), data[k, [0, 3]])
    assert np.allclose(y.numpy(), data[k, [4, 5]])
    assert mask.shape == (1, 64, 64)
