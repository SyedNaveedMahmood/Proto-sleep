import pytest
from mist_evidence.model import ModelConfig
from mist_cove.phase10.dependency_graph import support_map, interval_atoms


def cfg(**kw):
    return ModelConfig(**dict({'raw_control': True, 'radius': 0, 'use_crf': False}, **kw))


def test_historical_source_geometry():
    graph = support_map(cfg())
    assert [len(s['windows']) for s in graph['scales']] == [59, 57, 53]
    assert graph['feature_count'] == 209 and len(graph['global_features']) == 17
    assert graph['stage_order'] == ['Wake', 'N1', 'N2', 'N3', 'REM']
    for scale in graph['scales']:
        for w in scale['windows']:
            samples = set(range(w['start'], w['stop']))
            expected = [a for a in range(15) if samples.intersection(range(a*200, (a+1)*200))]
            assert w['atoms'] == expected
    assert all(g['atoms'] == list(range(15)) for g in graph['global_features'])


@pytest.mark.parametrize('start,stop,atoms', [(0,100,[0]),(150,250,[0,1]),(50,450,[0,1,2]),(2800,3000,[14])])
def test_off_grid_and_cross_boundary_support(start,stop,atoms):
    assert interval_atoms(start,stop) == atoms


def test_config_flags_and_rejection():
    assert support_map(cfg(use_amplitude=False))['feature_count'] == 208
    assert support_map(cfg(use_summary=False))['feature_count'] == 192
    for bad in [ModelConfig(), cfg(radius=10), cfg(use_crf=True), cfg(summary_only=True)]:
        with pytest.raises(ValueError): support_map(bad)
