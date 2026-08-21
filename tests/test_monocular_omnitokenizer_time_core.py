import gc
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OmniTokenizer import PostFusionOmniTokenizer, StereoOmniTokenizer, disparity_to_depth
from OmniTokenizer.omnitokenizer import (
    MonocularPosterior,
    OmniTokenizer_Decoder,
    OmniTokenizer_Encoder,
    TemporalOmniTokenizerDecoder,
    TemporalOmniTokenizerEncoder,
)


def _tiny_encoder():
    return TemporalOmniTokenizerEncoder(
        image_size=32,
        patch_embed="linear",
        norm_type="group",
        block="t",
        window_size=2,
        spatial_pos="none",
        patch_size=16,
        temporal_patch_size=4,
        spatial_depth=1,
        temporal_depth=1,
        dim=32,
        causal_in_peg=False,
        dim_head=8,
        heads=4,
        sequence_length=4,
    )


def _tiny_decoder():
    return TemporalOmniTokenizerDecoder(
        image_size=32,
        patch_embed="linear",
        norm_type="group",
        block="t",
        window_size=2,
        spatial_pos="none",
        patch_size=16,
        temporal_patch_size=4,
        spatial_depth=1,
        temporal_depth=1,
        dim=32,
        causal_in_peg=False,
        dim_head=8,
        heads=4,
        sequence_length=4,
        initial_disparity_px=8.0,
        disparity_scale=16.0,
    )


@pytest.fixture(scope="module")
def production_model():
    return StereoOmniTokenizer(initial_disparity_px=24.0, disparity_scale=32.0)


def test_structure_matches_post_fusion_design(production_model):
    assert PostFusionOmniTokenizer is StereoOmniTokenizer
    assert isinstance(production_model.encoder, OmniTokenizer_Encoder)
    assert isinstance(production_model.decoder, OmniTokenizer_Decoder)
    assert production_model.encoder.block == "ttww"
    assert production_model.decoder.block == "tttt"
    assert hasattr(production_model.decoder, "dec_temporal_transformer")
    assert hasattr(production_model.decoder, "temporal_unpatchify")
    assert not hasattr(production_model.decoder, "to_depth")

    encoder = production_model.encoder
    assert encoder.temporal_position_embedding.shape == (1, 4, 512)
    assert encoder.temporal_sampler.in_features == 4 * 512
    assert encoder.temporal_sampler.out_features == 512
    assert encoder.enc_temporal_transformer.layers[0][0] is None
    assert encoder.enc_temporal_transformer.layers[0][1].causal is False

    decoder = production_model.decoder
    assert decoder.temporal_position_embedding.shape == (1, 4, 512)
    assert decoder.temporal_unpatchify.out_features == 4 * 512
    assert decoder.dec_temporal_transformer.layers[0][0] is None
    assert decoder.dec_temporal_transformer.layers[0][1].causal is False
    # Video 模式已经拥有四个时间 token，最终 head 对每个 token 各生成一帧。
    assert decoder.to_pixels[0].out_features == 3 * 16 * 16
    assert decoder.to_disparity[0].out_features == 16 * 16
    assert decoder.to_pixels_first_frame[0].out_features == 3 * 16 * 16
    assert decoder.to_disparity_first_frame[0].out_features == 16 * 16


def test_spatial_encoder_stops_before_temporal_attention():
    encoder = _tiny_encoder().eval()
    calls = []
    hook = encoder.enc_temporal_transformer.register_forward_hook(
        lambda *_: calls.append("temporal")
    )
    try:
        features = encoder.encode_spatial_frames(
            torch.randn(6, 3, 4, 32, 32), is_image=False
        )
    finally:
        hook.remove()
    assert features.shape == (6, 4, 32, 2, 2)
    assert calls == []


def test_spatial_encoder_keeps_frames_independent():
    """Spatial PEG 的时间维必须为 1，修改一帧不能泄漏到其他帧。"""
    torch.manual_seed(0)
    encoder = _tiny_encoder().eval()
    video = torch.randn(1, 3, 4, 32, 32)
    changed = video.clone()
    # 只修改第 4 帧的局部 patch，避免整帧常数被 Patch LayerNorm 抵消。
    changed[:, :, 3, :4, :4] += 0.37

    with torch.no_grad():
        original = encoder.encode_spatial_frames(video, is_image=False)
        modified = encoder.encode_spatial_frames(changed, is_image=False)

    assert torch.equal(original[:, :3], modified[:, :3])
    assert not torch.equal(original[:, 3], modified[:, 3])


def test_spatial_transformers_receive_per_frame_video_shape():
    """Encoder/Decoder 的 Spatial PEG 都只能看到独立的单帧样本。"""
    encoder = _tiny_encoder().eval()
    decoder = _tiny_decoder().eval()
    encoder_shapes = []
    decoder_shapes = []

    def record_encoder_shape(_, args, kwargs):
        encoder_shapes.append((tuple(args[0].shape), kwargs['video_shape']))

    def record_decoder_shape(_, args, kwargs):
        decoder_shapes.append((tuple(args[0].shape), kwargs['video_shape']))

    hooks = [
        encoder.enc_spatial_transformer.register_forward_pre_hook(
            record_encoder_shape, with_kwargs=True
        ),
        decoder.dec_spatial_transformer.register_forward_pre_hook(
            record_decoder_shape, with_kwargs=True
        ),
    ]
    try:
        with torch.no_grad():
            encoder.encode_spatial_frames(torch.randn(1, 3, 4, 32, 32), is_image=False)
            decoder.decode_tokens(torch.randn(1, 32, 1, 2, 2), is_image=False)
    finally:
        for hook in hooks:
            hook.remove()

    assert encoder_shapes == [((4, 4, 32), (4, 1, 2, 2))]
    assert decoder_shapes == [((4, 4, 32), (4, 1, 2, 2))]


def test_post_fusion_temporal_order_and_view_isolation():
    torch.manual_seed(0)
    encoder = _tiny_encoder().eval()
    calls = []
    hooks = [
        encoder.enc_temporal_transformer.register_forward_hook(
            lambda *_: calls.append("attention")
        ),
        encoder.temporal_sampler.register_forward_hook(
            lambda *_: calls.append("sampler")
        ),
    ]
    fused = torch.randn(1, 3, 4, 32, 2, 2)
    changed = fused.clone()
    changed[:, 1] += 5
    try:
        with torch.no_grad():
            output = encoder.encode_temporal_fused(fused, is_image=False)
            modified = encoder.encode_temporal_fused(changed, is_image=False)
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == ["attention", "sampler", "attention", "sampler"]
    assert output.shape == (1, 3, 32, 1, 2, 2)
    assert torch.equal(output[:, 0], modified[:, 0])
    assert torch.equal(output[:, 2], modified[:, 2])
    assert not torch.equal(output[:, 1], modified[:, 1])


def test_encoder_temporal_attention_is_bidirectional_and_receives_t4():
    torch.manual_seed(0)
    encoder = _tiny_encoder().eval()
    attention_inputs = []
    attention_outputs = []

    def record_temporal(_, inputs, output):
        attention_inputs.append(tuple(inputs[0].shape))
        attention_outputs.append(output.detach())

    hook = encoder.enc_temporal_transformer.register_forward_hook(record_temporal)
    fused = torch.randn(1, 3, 4, 32, 2, 2)
    changed = fused.clone()
    changed[:, :, 3] += 5
    try:
        with torch.no_grad():
            encoder.encode_temporal_fused(fused, is_image=False)
            encoder.encode_temporal_fused(changed, is_image=False)
    finally:
        hook.remove()

    assert attention_inputs == [(12, 4, 32), (12, 4, 32)]
    # 修改最后一帧后，第一帧的 temporal feature 也应变化，证明没有 causal mask。
    assert not torch.equal(attention_outputs[0][:, 0], attention_outputs[1][:, 0])


def test_image_mode_skips_temporal_modules():
    encoder = _tiny_encoder().eval()
    calls = []
    hooks = [
        encoder.enc_temporal_transformer.register_forward_hook(
            lambda *_: calls.append("attention")
        ),
        encoder.temporal_sampler.register_forward_hook(
            lambda *_: calls.append("sampler")
        ),
    ]
    features = torch.randn(1, 3, 1, 32, 2, 2)
    try:
        output = encoder.encode_temporal_fused(features, is_image=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == []
    assert output.shape == (1, 3, 32, 1, 2, 2)
    assert torch.equal(output[:, :, :, 0], features[:, :, 0])


def test_decoder_order_is_unpatchify_temporal_then_spatial():
    decoder = _tiny_decoder().eval()
    calls = []
    hooks = [
        decoder.temporal_unpatchify.register_forward_hook(
            lambda *_: calls.append("unpatchify")
        ),
        decoder.dec_temporal_transformer.register_forward_hook(
            lambda *_: calls.append("temporal")
        ),
        decoder.dec_spatial_transformer.register_forward_hook(
            lambda *_: calls.append("spatial")
        ),
    ]
    tokens = torch.randn(1, 32, 1, 2, 2)
    try:
        with torch.no_grad():
            video_tokens = decoder.decode_tokens(tokens, is_image=False)
            assert calls == ["unpatchify", "temporal", "spatial"]
            calls.clear()
            image_tokens = decoder.decode_tokens(tokens, is_image=True)
            assert calls == ["spatial"]
    finally:
        for hook in hooks:
            hook.remove()

    assert video_tokens.shape == (1, 4, 2, 2, 32)
    assert image_tokens.shape == (1, 1, 2, 2, 32)


def test_decoder_heads_restore_video_and_image():
    decoder = _tiny_decoder().eval()
    tokens = torch.randn(2, 32, 1, 2, 2)
    with torch.no_grad():
        video = decoder.decode_rgb_disparity(tokens, is_image=False)
        image = decoder.decode_rgb_disparity(tokens, is_image=True)

    assert video[0].shape == (2, 3, 4, 32, 32)
    assert video[1].shape == video[2].shape == (2, 1, 4, 32, 32)
    assert image[0].shape == (2, 3, 1, 32, 32)
    assert image[1].shape == image[2].shape == (2, 1, 1, 32, 32)
    assert (video[1] > 0).all()
    assert torch.allclose(video[2], video[1] * 16.0)
    assert video[0].min() >= -1 and video[0].max() <= 1


def test_disparity_bias_and_depth_conversion():
    decoder = _tiny_decoder().eval()
    bias = decoder.to_disparity[0].bias
    initial = (F.softplus(bias) + decoder.disparity_epsilon) * decoder.disparity_scale
    assert torch.allclose(initial, torch.full_like(initial, 8.0), atol=1e-5)

    fx = torch.tensor([400.0, 500.0, 600.0])
    predicted = torch.full((2, 3, 1, 4, 8, 8), 20.0)
    depth = disparity_to_depth(predicted, fx=fx, baseline=0.08)
    expected = fx.view(1, 3, 1, 1, 1, 1) * 0.08 / predicted
    assert torch.equal(depth, expected)


def test_posterior_reuses_original_distribution():
    mean = torch.randn(2, 3, 4, 1, 2, 2)
    posterior = MonocularPosterior(mean, torch.zeros_like(mean))
    assert posterior.mean.shape == posterior.logvar.shape == mean.shape
    assert posterior.mode().shape == posterior.sample().shape == mean.shape
    assert posterior.kl().shape == (2,)


def test_tiny_forward_backward_reaches_post_fusion_modules():
    encoder = _tiny_encoder().train()
    decoder = _tiny_decoder().train()
    compressed = encoder.encode_temporal_fused(
        torch.randn(1, 3, 4, 32, 2, 2), is_image=False
    )
    rgb, normalized, _ = decoder.decode_rgb_disparity(
        compressed.reshape(3, 32, 1, 2, 2), is_image=False
    )
    loss = rgb.square().mean() + normalized.mean()
    loss.backward()

    for module in (
        encoder.enc_temporal_transformer,
        encoder.temporal_sampler,
        decoder.temporal_unpatchify,
        decoder.dec_temporal_transformer,
        decoder.dec_spatial_transformer,
        decoder.to_pixels,
        decoder.to_disparity,
    ):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H200 BF16 smoke requires CUDA")
def test_h200_bf16_post_fusion_forward_backward():
    device = torch.device("cuda")
    model = StereoOmniTokenizer(
        initial_disparity_px=24.0, disparity_scale=32.0
    ).to(device).train()
    fused = torch.randn(1, 3, 4, 512, 16, 16, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(fused, mode="video", sample_posterior=False)
        loss = output.rgb.square().mean() + output.normalized_disparity.mean()
    loss.backward()

    assert output.latent.shape == (1, 3, 48, 1, 16, 16)
    assert output.rgb.shape == (1, 3, 3, 4, 256, 256)
    assert output.disparity.shape == (1, 3, 1, 4, 256, 256)
    assert torch.isfinite(output.disparity).all() and (output.disparity > 0).all()

    for module in (
        model.encoder.enc_temporal_transformer,
        model.encoder.temporal_sampler,
        model.posterior_head,
        model.decoder.temporal_unpatchify,
        model.decoder.dec_temporal_transformer,
        model.decoder.dec_spatial_transformer,
        model.decoder.to_pixels,
        model.decoder.to_disparity,
    ):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = model(
            torch.randn(1, 3, 1, 512, 16, 16, device=device),
            mode="image",
            sample_posterior=False,
        )
    assert image.rgb.shape == (1, 3, 3, 1, 256, 256)
    assert image.disparity.shape == (1, 3, 1, 1, 256, 256)

    del image, output, fused, model
    gc.collect()
    torch.cuda.empty_cache()
