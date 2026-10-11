"""Training-step building-block classes for the laya perf patch (injected).

Same injection contract as ``core.laya_controls`` / ``core.laya_eval_fast`` /
``core.laya_train_fast``: cohesive classes, no module-level ``torch`` import
(the loop passes its local ``torch`` in), unit-tested on the host and exec'd
byte-for-byte into the staged kernels. Listed in
``cli.laya_train_patch_ddp.FINETUNE_CONTROL_LOGIC_SOURCE``.

These four were extracted VERBATIM from ``cli.laya_train_patch_loop``'s
``LOOP_PATCH_TEMPLATE`` to bring that template back under the 1k-line cap
(the same move that relocated ``DevEvaluator`` into ``core.laya_eval_fast``).
They are the pure per-step helpers the loop composes: the adversarial
perturber, the optimizer factory, the objective builder and the bf16-aware
forward wrapper. ``TrainingOptimizer`` references ``ParamGroupBuilder`` (a
``core.laya_controls`` class injected earlier), resolved at call time in the
shared kernel namespace.
"""
from __future__ import annotations

from typing import Any


class AdversarialPerturber:
    """FGM (embedding-direction) / AWP (weight-direction) perturbation.

    ``fgm`` perturbs the embedding parameters (the classic Fast Gradient
    Method input perturbation); ``awp`` perturbs every trainable parameter.
    Both step along the gradient direction and are restored after the extra
    forward/backward, so the recipe is unchanged when ``adv_eps == 0``.
    """

    @staticmethod
    def targets(model, params, adv_kind):
        if str(adv_kind).lower() != "fgm":
            return list(params)
        named = dict(model.named_parameters())
        picked = [param for name, param in named.items()
                  if "embed" in name.lower() and param.requires_grad]
        return picked or list(params)

    @staticmethod
    def perturb(torch: "Any", params: "list[Any]", adv_eps: float,  # noqa: UP037 -- injected without module imports
                norm: float) -> "list[tuple[Any, Any]]":  # noqa: UP037
        torch.clear_autocast_cache()
        deltas = []
        with torch.no_grad():
            for param in params:
                if param.grad is None:
                    continue
                delta = (adv_eps / norm) * param.grad.detach()
                param.add_(delta)
                deltas.append((param, delta))
        return deltas

    @staticmethod
    def restore(torch: "Any", deltas: "list[tuple[Any, Any]]") -> None:  # noqa: UP037 -- injected without module imports
        with torch.no_grad():
            for param, delta in deltas:
                param.sub_(delta)
        torch.clear_autocast_cache()


class TrainingOptimizer:
    """Optimizer construction + best-effort layer-wise LR decay."""

    @staticmethod
    def make(torch, model, groups, config, control):
        kind = str(control.get("optimizer") or "").lower()
        groups = ParamGroupBuilder.apply(  # noqa: F821 -- co-injected (core.laya_controls)
            model, groups, bool(control.get("no_decay_bias_norm")))
        fused = any(param.is_cuda for group in groups
                    for param in group["params"])
        if kind == "adafactor":
            try:
                return torch.optim.Adafactor(
                    groups, lr=config.head_lr,
                    weight_decay=config.weight_decay)
            except Exception:  # noqa: BLE001 - Adafactor missing -> AdamW
                print("[perf-patch] Adafactor unavailable; AdamW", flush=True)
        if kind == "lamb":
            try:
                from torch_optimizer import Lamb
                return Lamb(groups, lr=config.head_lr,
                            weight_decay=config.weight_decay)
            except Exception:  # noqa: BLE001 - torch_optimizer missing -> next
                try:
                    from transformers.optimization import Lamb
                    return Lamb(groups, lr=config.head_lr,
                                weight_decay=config.weight_decay)
                except Exception:  # noqa: BLE001 - Lamb missing -> AdamW
                    print("[perf-patch] Lamb unavailable; AdamW", flush=True)
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay,
                                 fused=fused)

    @staticmethod
    def apply_layer_decay(model: "Any", groups: "list[dict[str, Any]]",  # noqa: UP037 -- injected without module imports
                          layer_decay: float) -> "list[dict[str, Any]]":  # noqa: UP037
        import re as _re

        depths = {}
        for name, param in model.named_parameters():
            match = _re.search(r"(?:layers?|layer)[.]([0-9]+)[.]", name)
            if match:
                depths[id(param)] = int(match.group(1))
        if not depths:
            return groups
        deepest = max(depths.values())
        decayed = []
        for group in groups:
            base_lr = float(group["lr"])
            by_rate = {}
            for param in group["params"]:
                lr = base_lr * layer_decay ** (deepest - depths.get(id(param), deepest))
                by_rate.setdefault(lr, []).append(param)
            decayed.extend({**group, "params": parameters, "lr": lr}
                           for lr, parameters in by_rate.items())
        return decayed


class LossBuilder:
    """Class weighting, weighted soft-CE and per-row loss for hard mining."""

    @staticmethod
    def class_weights(torch, items, device):
        counts = {}
        for item in items:
            try:
                target = item["target"]
                label = max(range(len(target)),
                            key=lambda i: float(target[i]))
            except Exception:  # noqa: BLE001,S112 - a malformed item is skipped
                continue
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return None
        classes = max(counts) + 1
        total = sum(counts.values())
        weights = [0.0] * classes
        for label in range(classes):
            count = counts.get(label, 0)
            weights[label] = (total / (classes * count)) if count else 0.0
        return torch.tensor(weights, dtype=torch.float32, device=device)

    @staticmethod
    def weighted(torch, logits, target, mask, weights):
        logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
        per_row = -(target * logp * mask).sum(-1)
        labels = target.argmax(dim=-1)
        row_w = weights.to(logits.device)[labels]
        return (per_row * row_w).mean()

    @staticmethod
    def per_row(torch, logits, target, mask):
        logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
        return (-(target * logp * mask).sum(-1)).detach()

    @staticmethod
    def compute(torch, laya_train, config, logits, target, mask, qtype, sigma,
                w_sph, w_rps, class_weights, margin):
        """The objective for one forward: laya's rlcd / weighted / soft-CE.

        With no class weights and margin==0 this returns laya's
        ``soft_ce_loss`` unchanged (byte-identical default). The contrastive
        margin is applied to the soft-CE objective only (laya's ``rlcd_loss``
        exposes no margin); class_weight takes precedence over margin.
        """
        if config.loss == "rlcd":
            return laya_train.rlcd_loss(logits, target, mask, qtype, sigma,
                                        config.rl_samples, w_sph, w_rps)
        if margin and float(margin) > 0.0:
            logits = logits - float(margin) * (1.0 - target)
        if class_weights is not None:
            return LossBuilder.weighted(torch, logits, target, mask,
                                        class_weights)
        return laya_train.soft_ce_loss(logits, target, mask)


class Forwarder:
    """The bf16-aware forward wrapper (fp16 defers to laya's own _forward)."""

    @staticmethod
    def run(torch, laya_train, model, batch, device, amp, freeze_encoder,
            amp_dtype):
        if (not amp) or amp_dtype != "bf16" or not hasattr(torch, "autocast"):
            return laya_train._forward(model, batch, device, amp,
                                       freeze_encoder)
        args = (batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device))
        kwargs = {"detach_encoder": freeze_encoder}
        if "option_ids" in batch:
            kwargs.update(position_ids=batch["position_ids"].to(device),
                          option_ids=batch["option_ids"].to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, _act = model(*args, **kwargs)
        return logits.float()
