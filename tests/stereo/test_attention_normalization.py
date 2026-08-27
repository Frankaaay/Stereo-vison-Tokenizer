import unittest

import torch

from stereo_tokenizer.modules.attention import Attention


class AttentionNormalizationTest(unittest.TestCase):
    def _attention(self) -> Attention:
        return Attention(
            dim=8,
            dim_head=4,
            heads=2,
            dropout=0.0,
            spatial_pos="none",
        ).eval()

    def test_self_attention_uses_same_normalized_input_for_q_and_kv(self) -> None:
        attention = self._attention()
        x = torch.randn(2, 4, 8)
        x[..., 3] = x[..., 3] * 10_000.0 + 20_000.0
        captured = {}

        handles = [
            attention.to_q.register_forward_pre_hook(
                lambda _module, inputs: captured.setdefault("q", inputs[0].detach().clone())
            ),
            attention.to_kv.register_forward_pre_hook(
                lambda _module, inputs: captured.setdefault("kv", inputs[0].detach().clone())
            ),
        ]
        try:
            with torch.no_grad():
                attention(x, is_spatial=False)
        finally:
            for handle in handles:
                handle.remove()

        torch.testing.assert_close(captured["kv"], captured["q"])
        torch.testing.assert_close(captured["kv"], attention.norm(x))

    def test_cross_attention_keeps_context_normalization(self) -> None:
        attention = self._attention()
        x = torch.randn(2, 4, 8)
        context = torch.randn(2, 6, 8)
        context[..., 5] = context[..., 5] * 1_000.0
        captured = {}

        handle = attention.to_kv.register_forward_pre_hook(
            lambda _module, inputs: captured.setdefault("kv", inputs[0].detach().clone())
        )
        try:
            with torch.no_grad():
                attention(x, context=context, is_spatial=False)
        finally:
            handle.remove()

        torch.testing.assert_close(captured["kv"], attention.context_norm(context))


if __name__ == "__main__":
    unittest.main()
