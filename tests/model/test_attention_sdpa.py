import unittest

import torch
import torch.nn.functional as F

from stereo_tokenizer.modules.attention import Attention


class AttentionSDPATest(unittest.TestCase):
    def test_relative_bias_sdpa_matches_fallback_without_dropout(self):
        torch.manual_seed(3)
        attention = Attention(
            dim=8, dim_head=4, heads=2, dropout=0.0, spatial_pos="rel"
        ).eval()
        x = torch.randn(2, 4, 8)
        with torch.no_grad():
            expected = attention(x)
            sdpa = F.scaled_dot_product_attention
            delattr(F, "scaled_dot_product_attention")
            try:
                actual = attention(x)
            finally:
                F.scaled_dot_product_attention = sdpa
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_relative_bias_receives_gradient_through_sdpa(self):
        attention = Attention(
            dim=8, dim_head=4, heads=2, dropout=0.0, spatial_pos="rel"
        ).train()
        attention(torch.randn(2, 4, 8)).square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in attention.spatial_rel_pos_bias.parameters()
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertGreater(
            sum(
                torch.count_nonzero(gradient).item()
                for gradient in gradients
                if gradient is not None
            ),
            0,
        )

    def test_eval_disables_attention_dropout(self):
        attention = Attention(
            dim=8, dim_head=4, heads=2, dropout=0.75, spatial_pos="rel"
        ).eval()
        x = torch.randn(2, 4, 8)
        with torch.no_grad():
            first = attention(x)
            second = attention(x)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_padding_causal_and_null_kv_do_not_leak_masked_token(self):
        torch.manual_seed(9)
        attention = Attention(
            dim=8,
            dim_head=4,
            heads=2,
            causal=True,
            num_null_kv=1,
            dropout=0.0,
            spatial_pos="rel",
        ).eval()
        baseline = torch.randn(1, 4, 8)
        changed = baseline.clone()
        changed[:, 3] += 1000
        mask = torch.tensor([[True, True, True, False]])
        with torch.no_grad():
            baseline_output = attention(baseline, mask=mask)
            changed_output = attention(changed, mask=mask)
        torch.testing.assert_close(
            baseline_output[:, :3], changed_output[:, :3], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
