"""In-memory random historical architectures, never fitted source artifacts."""
from dataclasses import dataclass
import torch
from mist_evidence.model import ModelConfig
from mist_transfer.baselines import make_baseline


@dataclass(frozen=True)
class MockInstance:
    network: object
    seed: int
    origin: str = 'GENERATED_IN_MEMORY_NO_SOURCE_ARTIFACT'


def make_mock(seed=123, *, use_summary=True, use_amplitude=True):
    if seed not in (123, 456, 789):
        raise ValueError('only frozen mock seeds allowed')
    cfg = ModelConfig(radius=0, use_crf=False, use_summary=use_summary, use_amplitude=use_amplitude)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        waves = [torch.randn(cfg.prototypes_per_scale, scale) for scale in cfg.scales]
        bank = {'config': cfg.dictionary(), 'waveforms': waves, 'amplitude_stats': (0., 1.)}
        model = make_baseline('raw_context', bank, {'mean_uV': 0., 'std_uV': 1.}).eval()
    return MockInstance(model, seed)


def mock_input(seed=901, n=3):
    if not isinstance(n, int) or not 1 <= n <= 8:
        raise ValueError('bounded synthetic batch required')
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(3000) / 100
    x = torch.randn(n, 1, 3000, generator=gen) * .15
    for i in range(n):
        x[i, 0] += (i+1)*.2*torch.sin(2*torch.pi*(i+2)*t) + .07*i
    return x
