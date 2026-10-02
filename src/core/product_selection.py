"""Separate an explicitly selected SKU from the title's list of choices."""
from __future__ import annotations

import re

from core.text import normalized_attribute_text


def selected_product_title(title: object) -> tuple[str, str]:
    text = str(title or '')
    choices = re.search(r'\b(?:your choice of|choose from|available in)\b', text, re.I)
    selection = re.search(r'\(([^()]*)\)\s*$', text)
    if not choices or not selection or selection.start() < choices.end():
        return text, ''
    value = selection.group(1)
    # A bare pack/size suffix cannot establish which option was selected.
    value = re.sub(r'\b(?:pack|set|case|bundle)\s+of\s+\d+\b', '', value, flags=re.I)
    value = re.sub(r'\b\d+(?:[.,]\d+)?\s*(?:bottles?|cans?|packs?|ct|count|ml|cl|l|fl\s*oz|oz)\b', '', value, flags=re.I)
    variant = normalized_attribute_text(value).strip()
    if not variant or variant in {'pack', 'bottle', 'can', 'size'}:
        return text, ''
    return text[:choices.start()] + ' ' + value, variant


def selected_identity_inputs(title: object, attributes: object) -> tuple[str, str, str]:
    selected, variant = selected_product_title(title)
    attrs = str(attributes or '')
    if variant:
        # The captured flavor field may itself enumerate the choices. Retain
        # the raw field in source rows; it cannot override explicit selection.
        attrs = ';'.join(part for part in attrs.split(';')
                         if not re.match(r'\s*flavou?r\s*:', part, re.I))
    return selected, attrs, variant
