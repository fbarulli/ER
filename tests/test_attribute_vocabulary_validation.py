"""Standalone extraction and common loading share fail-closed vocabulary checks."""
import copy
import json

import pytest

from core.attribute_vocabulary import validated_attribute_vocabulary
from core.common import TRAIN_ROOT, _read_vocabulary
from core import critical_attributes


@pytest.mark.parametrize('invalid', ['missing', 'empty_lexicon', 'blank_token', 'unknown_alias'])
def test_both_readers_reject_unusable_attribute_vocabulary(tmp_path, monkeypatch, invalid):
    data = json.loads((TRAIN_ROOT / 'config/vocabulary.json').read_text())
    if invalid == 'missing':
        data.pop('attribute_vocabulary')
    elif invalid == 'empty_lexicon':
        data['attribute_vocabulary']['made_from_lexicon'] = []
    elif invalid == 'blank_token':
        data['attribute_vocabulary']['caffeine_sources'] = [' ']
    else:
        data['attribute_vocabulary']['flavor_aliases']['example'] = 'not in lexicon'
    (tmp_path / 'config').mkdir()
    path = tmp_path / 'config/vocabulary.json'
    path.write_text(json.dumps(data))
    from core.project_root import ProjectRoot
    monkeypatch.setattr(ProjectRoot, 'find', lambda _: tmp_path)
    critical_attributes._attribute_vocabulary.cache_clear()
    try:
        with pytest.raises(SystemExit, match='vocabulary.attribute_vocabulary'):
            critical_attributes._attribute_vocabulary()
        with pytest.raises(SystemExit, match='vocabulary.attribute_vocabulary'):
            _read_vocabulary(path)
    finally:
        critical_attributes._attribute_vocabulary.cache_clear()


def test_validation_preserves_config_values_without_mutation():
    data = json.loads((TRAIN_ROOT / 'config/vocabulary.json').read_text())
    before = copy.deepcopy(data)
    assert validated_attribute_vocabulary(data) == data['attribute_vocabulary']
    assert data == before
