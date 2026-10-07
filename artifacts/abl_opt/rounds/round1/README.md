# Round 1: optimize save_report (CURRENT #1)
## target
`_saved_track_report` calls `save_report(...)` which **re-serializes `validated` twice** per track
(dashboard pointer + `<track>/ablation/report.json`) — even in complete_saved's trusted-cache
restore path where the serialized bytes ALREADY EXIST on disk (round 0 measurement: they ARE
written as a legal side effect every round). Identical round-left state arrives when:
 - validated payload is identical to the previous round's untouched file bytes, and
 - a legal served artifact exists at the dashboard pointer
## saving rule
When the target report.json already exists AND is byte-identical to the round's serialized
document, skip the byte-identical writes. Detect byte-identity by a one-shot SHA-256 of the
existing file vs the digest of the round's would-be-serialised bytes, computed WITHOUT writing.
This avoids redundant disk I/O and redundant serialization while PRESERVING the side effect
exactness: if the round left a different report (recompute path, settings drift), the write
happens every time as before.
## byte-identity verification
- `artifacts/abl_opt/rounds/round1/report_bytes_before.json` — the "before" receipt+report payloads
- `artifacts/abl_opt/rounds/round1/report_bytes_after.json` — the "after" receipt+report payloads
- diff of these two files must be EMPTY (byte-identical post-state).
