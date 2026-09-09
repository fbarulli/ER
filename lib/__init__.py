"""lib — shared, import-light utilities (config SSOT, loaders, NLP).

Package marker only. lib/__init__ stays deliberately empty of imports: a
module-level `from TRAIN... import` here would make EVERY `import lib.*`
drag in TRAIN.training → torch plus a full 53MB dataset sha256 at import
time (the dataset-tag print was the live symptom of exactly that bug).
"""
