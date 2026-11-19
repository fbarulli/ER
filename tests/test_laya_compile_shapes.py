"""Compiled HPO forwards must not specialize unrelated equal dimensions."""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch

from core.laya_train_fast import ModelCompiler


class ShapeModel(torch.nn.Module):
    def forward(self, tokens: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        return tokens.sin().sum(dim=-1, keepdim=True) + markers.cos()


class CountingCompiler:
    def __init__(self) -> None:
        self.graphs = 0

    def backend(self, graph: torch.fx.GraphModule, inputs: list[Any]) -> Callable[..., Any]:
        self.graphs += 1
        return graph.forward

    def compile(self, model: torch.nn.Module, *, mode: str, dynamic: bool) -> torch.nn.Module:
        return torch.compile(model, backend=self.backend, dynamic=dynamic)


class TestCompileShapes:
    def test_older_laya_keeps_compilation_and_reports_missing_shape_helper(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setitem(sys.modules, "laya._compile", None)
        model = ShapeModel()
        compiled = ModelCompiler.compile(CountingCompiler(), model, "default")
        inputs = (torch.ones(2, 3), torch.ones(2, 4))
        assert compiled is not model
        torch.testing.assert_close(compiled(*inputs), model(*inputs))
        output = capsys.readouterr().out
        assert "Traceback (most recent call last)" in output
        assert "keeping ordinary dynamic compilation" in output

    def test_variable_dimensions_reuse_one_graph_and_preserve_gradients(self) -> None:
        torch._dynamo.reset()
        compiler = CountingCompiler()
        model = ShapeModel()
        compiled = ModelCompiler.compile(compiler, model, "default")
        for rows, tokens, markers in ((4, 4, 4), (3, 7, 2), (5, 9, 3)):
            inputs = (torch.randn(rows, tokens, requires_grad=True),
                      torch.randn(rows, markers, requires_grad=True))
            expected = model(*inputs)
            result = compiled(*inputs)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            expected_grad = torch.autograd.grad(expected.sum(), inputs)
            result_grad = torch.autograd.grad(result.sum(), inputs)
            for left, right in zip(expected_grad, result_grad, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert compiler.graphs == 1
