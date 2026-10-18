"""Predeclared known-component fixtures; no dataset imports or fitted statistics."""
from dataclasses import dataclass
import torch

from mist_cove.synthetic import FREQUENCIES


@dataclass
class Fixture:
    x: torch.Tensor
    label: int
    diagnostic_intervals: list[tuple[int, int]]
    donor_interval: tuple[int, int]
    component_names: list[str]
    case: str
    task: str


def make_fixtures(n, seed, task, case, generator):
    if task not in {"substitutable", "complementary"} or case not in {"single", "dual", "redundant", "correlated", "distractor", "competition", "overlap", "ties"}:
        raise ValueError("unregistered synthetic task/case")
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(200) / 100
    envelope = torch.hann_window(200, periodic=False)
    rows = []
    for i in range(n):
        label = i % 5
        x = torch.randn(1, 1, 3000, generator=g) * generator["noise_std_uV"]
        slots = (torch.randperm(13, generator=g) + 1) * 200
        amplitude = generator["amplitude_uV"][0] + (generator["amplitude_uV"][1] - generator["amplitude_uV"][0]) * float(torch.rand((), generator=g))
        fa = FREQUENCIES[label][0]
        fb = FREQUENCIES[label][1] if task == "substitutable" else FREQUENCIES[(label + 1) % 5][0]
        phase_a = 2 * torch.pi * torch.rand((), generator=g)
        phase_b = 2 * torch.pi * torch.rand((), generator=g)
        a = amplitude * envelope * torch.sin(2 * torch.pi * fa * t + phase_a)
        b = amplitude * envelope * torch.sin(2 * torch.pi * fb * t + phase_b)
        if (generator["version"] == "known-components-phase2-v2" and task == "complementary"
                and case in {"single", "redundant", "correlated", "distractor", "ties"}):
            # The surviving role is unobserved and randomized independently of
            # label. v1's fixed-A removal policy did not enforce this ambiguity.
            if bool(torch.randint(2, (), generator=g)):
                a = b
        if case in {"single", "distractor"}:
            waves = [a]
        elif case == "redundant":
            waves = [a, a.clone()]
        elif case == "correlated":
            waves = [a + .02 * torch.randn(200, generator=g) for _ in range(generator["correlated_copies"])]
        elif case == "ties":
            # Exact repeated physical patches: no hidden background-rank jitter.
            waves = [a.clone() for _ in range(8)]
        else:
            waves = [a, b]
        intervals = []
        occupied = []
        for j, wave in enumerate(waves):
            start = int(slots[j])
            if case == "overlap" and j == 1:
                start = intervals[0][0] + 100
            if case == "ties":
                x[0, 0, start:start + 200] = wave
            else:
                x[0, 0, start:start + 200] += wave
            intervals.append((start, start + 200)); occupied.append((start, start + 200))
        if case in {"distractor", "competition"}:
            count = 1 if case == "distractor" else 4
            for j in range(count):
                start = int(slots[len(waves) + j])
                freq = generator["artifact_frequency_hz"] if case == "distractor" else FREQUENCIES[int(torch.randint(5, (), generator=g))][0]
                nuisance = amplitude * generator["artifact_amplitude_multiplier"] * envelope * torch.sin(2 * torch.pi * freq * t + 2 * torch.pi * torch.rand((), generator=g))
                x[0, 0, start:start + 200] += nuisance
                occupied.append((start, start + 200))
        donor = next((s, s + 200) for s in range(0, 3000, 200)
                     if all(s + 200 <= lo or s >= hi for lo, hi in occupied))
        rows.append(Fixture(x, label, intervals, donor, [f"diagnostic_{j}" for j in range(len(intervals))], case, task))
    return rows


def training_batch(n, seed, task, generator):
    # All methods see identical examples and order within each task/seed.
    rows = make_fixtures(n, seed, task, "dual", generator)
    if task == "substitutable":
        singles = make_fixtures(n, seed + 71, task, "single", generator)
        rows = [single if i % 3 == 0 else dual for i, (single, dual) in enumerate(zip(singles, rows))]
    return torch.cat([row.x for row in rows]), torch.tensor([row.label for row in rows])


def transformed(x, name, settings, seed):
    g = torch.Generator().manual_seed(seed)
    if name == "noise":
        return x + torch.randn(x.shape, generator=g).to(x.device) * settings["noise_std_uV"]
    if name == "shift":
        d = settings["shift_samples"]
        return torch.cat((torch.zeros_like(x[..., :d]), x[..., :-d]), -1)
    if name == "physical_gain":
        return x * settings["physical_gain"]
    if name == "calibrated_gain":
        gain, offset = settings["calibration_gain"], settings["calibration_offset_uV"]
        return ((x * gain + offset) - offset) / gain
    raise ValueError("unknown controlled transformation")
