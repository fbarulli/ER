"""Declared quantities share title measurement grammar and whole count tokens."""
import pytest
from pipeline import parse_attribute_volume_pack

@pytest.mark.parametrize('text,volume',[('Volume: 1, 5 l',1500.),
                                      ('Volume: 1 / 2 l',500.),
                                      ('Volume: 1.5 dl',150.),
                                      ('Volume: 355',355.),
                                      ('Volume: 355 ml; Count per Unit: 6',355.),
                                      ('Volume: 355 ml Count per Unit: 6',355.)])
def test_declared_volume_uses_complete_measurement(text,volume):
    result=parse_attribute_volume_pack(text)
    assert result[:2] == (volume,.9)

@pytest.mark.parametrize('count',['2.5','2,5','2/5','0'])
def test_fractional_or_zero_declared_count_is_unknown(count):
    assert parse_attribute_volume_pack('Count per Unit: '+count)[2:] == (1,0.)


def test_valid_declared_count_is_preserved():
    quantity,confidence=parse_attribute_volume_pack('Count per Unit: 24')[2:]
    assert quantity==24 and confidence>0


@pytest.mark.parametrize('text',['Volume: 0','Volume: 0 ml','Volume: 0,0'])
def test_zero_declared_volume_stays_unknown(text):
    assert parse_attribute_volume_pack(text)[:2] == (0.,0.)
