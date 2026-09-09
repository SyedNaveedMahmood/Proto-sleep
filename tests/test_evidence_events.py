import pytest
from protosleep.evidence.audit import event_scores


def event(a, b, recording="night1", kind="spindle"):
    return {"start_s": a, "end_s": b, "recording": recording, "event": kind}


def test_event_matching_not_double_counted():
    result = event_scores([event(1,2), event(1,2)], [event(1,2)], .5)
    assert (result["tp"], result["fp"], result["fn"]) == (1,1,0)


def test_event_types_and_recordings_are_separate():
    result = event_scores([event(1,2,"night2"), event(1,2,kind="K-complex")], [event(1,2)], .5)
    assert result["tp"] == 0


def test_touching_intervals_do_not_overlap():
    assert event_scores([event(1,2)], [event(2,3)], .1)["tp"] == 0


def test_unreviewed_labels_cannot_be_scored():
    with pytest.raises(ValueError):
        event_scores([event(1,2,kind="unreviewed")], [event(1,2)], .5)
