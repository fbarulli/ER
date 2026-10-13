"""The ablation lane's declared identity: the declarations themselves.

Every name here answers one question -- *which declared thing is this?* -- from
the declaration itself. Owner directive 2026-10-08: identity in this repository
is STRUCTURAL. A content digest is forbidden, and a byte length is not an
identity at all: it aliases every distinct value of the same size, which is how
two different ablation requests came to share one staging folder, two different
graph removals came to share one inference job (and one tensor set), and two
different cohorts came to compare equal.

``model_tracks.ablation`` imports these names by their historical spellings, so
the lane keeps ONE identity surface while this module holds its implementation.
"""
from __future__ import annotations

import json

from graph_tracks.data import NUMERIC, RELATIONS


def digest(value: object) -> str:
    """The structural identity of one declared value: its own canonical text.

    ``json.dumps(sort_keys=True)`` renders the value's DECLARED parameters, in
    canonical key order with JSON types preserved, so two different declared
    values can never share an identity. Deliberately neither a content digest
    nor a byte length (owner directive 2026-10-08): a byte length aliases every
    value of the same size, and callers compose this identity into array keys
    and file names, so it stays text.

    shortcut: the identity is only as small as the value it names, so this is
    called with DECLARED parameters (slice axes, support/vocabulary, native
    texts), never with a whole prepared request; upgrade if a caller needs a
    value that cannot be spelled structurally.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def cohort_identity(chosen: list[dict]) -> str:
    """The structural identity of one selected ablation cohort.

    Every pair's declared ``(sku_id1, sku_id2, label)`` in cohort order, rendered
    canonically. These are the values themselves, never a byte length or a
    content digest (owner directive 2026-10-08), so two different cohorts can
    never alias by coincidence of size; every track of one suite compares this
    identity to prove it ablates exactly the same cohort.
    """
    return digest([{key: pair[key] for key in ('sku_id1', 'sku_id2', 'label')}
                   for pair in chosen])


def request_folder(request: dict) -> str:
    """The DECLARED name of one ablation request's staging folder.

    A staging folder holds one request per ``(track, checkpoint role)`` in its
    output directory, so it is named by those declared parameters -- never by a
    content digest or a byte length (owner directive 2026-10-08). Re-preparing
    the same declaration rebuilds its tensors in place, which is the behaviour
    the content-addressed folder had. The name is deliberately never a track name
    alone: the same output directory holds the fixed per-track templates.
    """
    return f"{request['track']}_{request['checkpoint_role']}"


def graph_field_target(field: str) -> tuple[str, str]:
    """One declared ``channel.key`` graph field -> its validated (channel, key).

    The ONE place a declared graph field is split and checked against its
    channel's allowed keys, shared by the intervention that removes it and the
    identity that names the record set the removal derives, so a field can never
    be parsed two ways.
    """
    channel, key = field.split('.', 1)
    allowed = RELATIONS if channel == 'attribute' else NUMERIC if channel == 'numeric' else ()
    if key not in allowed:
        raise ValueError(f'unsupported graph field: {field}')
    return channel, key


def occurring_graph_fields(fields, records) -> tuple[str, ...]:
    """The declared graph fields a record set actually carries, sorted.

    ``graph_removed`` pops each declared channel key, so a declared field that no
    record carries is a no-op. The SET of occurring fields is therefore the exact
    structural identity of the derived record set -- never a byte length, which
    aliases two different removals of the same size (owner directive
    2026-10-08) -- and two attributes declaring the same occurring set derive the
    same records.
    """
    def occurs(field: str) -> bool:
        channel, key = graph_field_target(field)
        return any(key in record.get(channel, {}) for record in records)
    return tuple(sorted(field for field in fields if occurs(field)))
