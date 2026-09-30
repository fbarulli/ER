"""How reliable are product dimensions when listings share a valid GTIN?

Finding 01: 5,657 of 19,123 sampled cross-retailer same-GTIN pairs have
at least one disjoint raw dimension. These are evidence disagreements,
not confirmed different products. Inspect original listings and images
before deciding whether the problem is metadata, units, packaging, or identity.

Evidence originates from scripts/audit_identity_dimensions.py and subsequent
read-only checks against all 13 columns of dataset.csv.
"""
