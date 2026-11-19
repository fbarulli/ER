"""Embedded HPO Kaggle kernel-script text (cli.laya_hpo).

The in-source body ``cli.laya_hpo`` token-substitutes and stages as the HPO
Kaggle kernel script. One literal, split head/tail so no module crosses the
1k-line limit (mirrors ``cli.laya_kernel_text_edge``).
"""
from __future__ import annotations

from cli.laya_hpo_kernel_text_head import HPO_KERNEL_TEMPLATE_HEAD
from cli.laya_hpo_kernel_text_tail import HPO_KERNEL_TEMPLATE_TAIL

HPO_KERNEL_TEMPLATE = HPO_KERNEL_TEMPLATE_HEAD + HPO_KERNEL_TEMPLATE_TAIL
