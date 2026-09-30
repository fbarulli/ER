"""Which apparent conflicts require measurement or packaging context?

Finding 03: raw caffeine mg values lack a structured denominator. Multipack
text frequently lacks Count per Unit. Packaging attributes may describe an
inner container or outer wrapping. Brand strings may describe different levels.

Reproduce with PYTHONPATH=src .venv/bin/python scripts/audit_identity_context.py.
Counts cover original listings and must not be compared as percentages of the
capped deduped pair population in finding 01. Signals are review candidates.
"""
