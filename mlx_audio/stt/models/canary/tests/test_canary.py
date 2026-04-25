import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.canary.canary import CanaryEncoder, Model
from mlx_audio.stt.models.canary.config import (
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    PreprocessorConfig,
)
from mlx_audio.stt.models.canary.decoder import (
    CanaryDecoder,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    TransformerDecoderBlock,
)


def _small_encoder_config():
    return EncoderConfig(
        feat_in=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ff_expansion_factor=2,
        subsampling_factor=2,
        self_attention_model="rel_pos",
        subsampling="dw_striding",
        conv_kernel_size=3,
        subsampling_conv_channels=16,
        pos_emb_max_len=256,
        xscaling=True,
    )


def _small_decoder_config():
    return DecoderConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        inner_size=64,
    )


def _small_model_config():
    return ModelConfig(
        model_type="canary",
        preprocessor=PreprocessorConfig(features=16),
        encoder=_small_encoder_config(),
        transf_decoder=_small_decoder_config(),
        vocab_size=64,
        enc_output_dim=32,
    )


class TestConfig(unittest.TestCase):

    def test_preprocessor_config_defaults(self):
        config = PreprocessorConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.features, 128)
        self.assertEqual(config.normalize, "per_feature")

    def test_preprocessor_win_hop_length(self):
        config = PreprocessorConfig(
            sample_rate=16000, window_size=0.025, window_stride=0.01
        )
        self.assertEqual(config.win_length, 400)
        self.assertEqual(config.hop_length, 160)

    def test_encoder_config_from_dict(self):
        d = {"feat_in": 128, "n_layers": 32, "d_model": 1024, "n_heads": 8}
        config = EncoderConfig.from_dict(d)
        self.assertEqual(config.n_layers, 32)
        self.assertEqual(config.d_model, 1024)

    def test_decoder_config_from_dict(self):
        d = {
            "num_layers": 8,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "inner_size": 4096,
        }
        config = DecoderConfig.from_dict(d)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.inner_size, 4096)

    def test_decoder_config_from_nested_dict(self):
        d = {
            "decoder": {
                "num_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "inner_size": 4096,
            }
        }
        config = DecoderConfig.from_dict(d)
        self.assertEqual(config.num_layers, 8)

    def test_model_config_from_dict(self):
        d = {
            "model_type": "canary",
            "preprocessor": {"features": 128, "sample_rate": 16000},
            "encoder": {"feat_in": 128, "n_layers": 32, "d_model": 1024, "n_heads": 8},
            "transf_decoder": {
                "num_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "inner_size": 4096,
            },
            "vocab_size": 16384,
            "enc_output_dim": 1024,
        }
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.vocab_size, 16384)
        self.assertIsInstance(config.encoder, EncoderConfig)
        self.assertIsInstance(config.transf_decoder, DecoderConfig)

    def test_model_config_ignores_extra_keys(self):
        d = {
            "model_type": "canary",
            "unknown_field": 42,
            "preprocessor": {"features": 128, "extra": True},
        }
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.model_type, "canary")


class TestMultiHeadSelfAttention(unittest.TestCase):

    def test_output_shape(self):
        attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x = mx.random.normal((1, 10, 32))
        out, cache = attn(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))

    def test_cache_update(self):
        attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x1 = mx.random.normal((1, 5, 32))
        out1, cache1 = attn(x1)
        mx.eval(out1)

        x2 = mx.random.normal((1, 1, 32))
        out2, cache2 = attn(x2, cache=cache1)
        mx.eval(out2)

        self.assertEqual(cache2[0].shape[2], 6)  # 5 + 1


class TestMultiHeadCrossAttention(unittest.TestCase):

    def test_output_shape(self):
        attn = MultiHeadCrossAttention(d_model=32, n_heads=4)
        x = mx.random.normal((1, 5, 32))
        enc = mx.random.normal((1, 20, 32))
        out, cache = attn(x, enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 32))

    def test_cache_reuse(self):
        attn = MultiHeadCrossAttention(d_model=32, n_heads=4)
        enc = mx.random.normal((1, 20, 32))

        x1 = mx.random.normal((1, 5, 32))
        out1, cache1 = attn(x1, enc)
        mx.eval(out1)

        x2 = mx.random.normal((1, 1, 32))
        out2, cache2 = attn(x2, enc, cache=cache1)
        mx.eval(out2)
        self.assertEqual(out2.shape, (1, 1, 32))


class TestTransformerDecoderBlock(unittest.TestCase):

    def test_output_shape(self):
        block = TransformerDecoderBlock(d_model=32, n_heads=4, inner_size=64)
        x = mx.random.normal((1, 5, 32))
        enc = mx.random.normal((1, 20, 32))
        out, self_cache, cross_cache = block(x, enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 32))


class TestCanaryDecoder(unittest.TestCase):

    def test_output_shape(self):
        config = _small_decoder_config()
        decoder = CanaryDecoder(config, vocab_size=64, d_model=32)
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        enc = mx.random.normal((1, 20, 32))
        logits, cache = decoder(tokens, enc)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 3, 64))
        self.assertEqual(len(cache), 2)

    def test_autoregressive_step(self):
        config = _small_decoder_config()
        decoder = CanaryDecoder(config, vocab_size=64, d_model=32)
        enc = mx.random.normal((1, 20, 32))

        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        logits, cache = decoder(prompt, enc, start_pos=0)
        mx.eval(logits)

        next_token = mx.array([[4]], dtype=mx.int32)
        logits2, cache2 = decoder(next_token, enc, cache=cache, start_pos=3)
        mx.eval(logits2)
        self.assertEqual(logits2.shape, (1, 1, 64))


class TestModelSanitize(unittest.TestCase):

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_encoder_key_mapping(self):
        weights = {"encoder.layers.0.self_attn.linear_q.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("encoder.conformer.layers.0.self_attn.linear_q.weight", sanitized)

    def test_decoder_embedding_mapping(self):
        weights = {
            "transf_decoder._embedding.token_embedding.weight": mx.zeros((64, 32))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.embedding.weight", sanitized)

    def test_decoder_position_mapping(self):
        weights = {
            "transf_decoder._embedding.position_embedding.pos_enc": mx.zeros((1024, 32))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.position_embedding.pos_enc", sanitized)

    def test_decoder_layer_self_attn_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.first_sub_layer.query_net.weight": mx.zeros(
                (32, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.self_attn.q_proj.weight", sanitized)

    def test_decoder_layer_cross_attn_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.second_sub_layer.key_net.weight": mx.zeros(
                (32, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.cross_attn.k_proj.weight", sanitized)

    def test_decoder_layer_ffn_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.third_sub_layer.dense_in.weight": mx.zeros(
                (64, 32)
            )
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.ff1.weight", sanitized)

    def test_decoder_layer_norm_mapping(self):
        weights = {
            "transf_decoder._decoder.layers.0.layer_norm_1.weight": mx.zeros((32,)),
            "transf_decoder._decoder.layers.0.layer_norm_2.weight": mx.zeros((32,)),
            "transf_decoder._decoder.layers.0.layer_norm_3.weight": mx.zeros((32,)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.blocks.0.self_attn_norm.weight", sanitized)
        self.assertIn("decoder.blocks.0.cross_attn_norm.weight", sanitized)
        self.assertIn("decoder.blocks.0.ff_norm.weight", sanitized)

    def test_decoder_final_norm_mapping(self):
        weights = {"transf_decoder._decoder.final_layer_norm.weight": mx.zeros((32,))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.final_norm.weight", sanitized)

    def test_output_proj_mapping(self):
        weights = {"log_softmax.mlp.layer0.weight": mx.zeros((64, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.output_proj.weight", sanitized)

    def test_skips_dropout_keys(self):
        weights = {
            "transf_decoder._decoder.layers.0.first_sub_layer.attn_dropout.p": mx.array(
                [0.1]
            ),
            "transf_decoder._decoder.layers.0.first_sub_layer.layer_dropout.p": mx.array(
                [0.1]
            ),
        }
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 0)

    def test_conv1d_transpose(self):
        weights = {
            "encoder.layers.0.conv.pointwise_conv1.weight": mx.zeros((64, 32, 1))
        }
        sanitized = self.model.sanitize(weights)
        key = "encoder.conformer.layers.0.conv.pointwise_conv1.weight"
        self.assertEqual(sanitized[key].shape, (64, 1, 32))

    def test_conv2d_transpose(self):
        weights = {"encoder.pre_encode.conv.0.weight": mx.zeros((256, 1, 3, 3))}
        sanitized = self.model.sanitize(weights)
        key = "encoder.conformer.pre_encode.conv.0.weight"
        self.assertEqual(sanitized[key].shape, (256, 3, 3, 1))

    def test_skips_encoder_decoder_proj(self):
        weights = {"encoder_decoder_proj.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 0)

    # Pointwise-conv regression cases. PyTorch (out, in, *kernel) shapes
    # whose trailing dim is small (1, etc.) are easy to misclassify by
    # any shape-only heuristic. With sentinel-key-based detection these
    # must transpose to MLX layout because no MLX-converted sentinel
    # appears in the dict.
    def test_conv1d_pointwise_transpose(self):
        weights = {"encoder.x.conv.pointwise.weight": mx.zeros((64, 32, 1))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(
            sanitized["encoder.conformer.x.conv.pointwise.weight"].shape, (64, 1, 32)
        )

    def test_conv2d_pointwise_transpose(self):
        weights = {"encoder.x.conv.weight": mx.zeros((256, 256, 1, 1))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(
            sanitized["encoder.conformer.x.conv.weight"].shape, (256, 1, 1, 256)
        )

    def test_conv2d_kw1_transpose(self):
        # PyTorch first conv with kernel (3, 1): (out=64, in=1, kH=3, kW=1).
        weights = {"encoder.x.conv.weight": mx.zeros((64, 1, 3, 1))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(
            sanitized["encoder.conformer.x.conv.weight"].shape, (64, 3, 1, 1)
        )


class TestModelSanitizeMLXConverted(unittest.TestCase):
    """Sanitize support for canary-mlx-converted canary checkpoints.

    The only public converter producing MLX-format canary weights is
    canary-mlx (https://github.com/QuentinFuxa/canary-mlx); public
    examples include `eelcor/canary-1b-v2-mlx` and
    `Mediform/canary-1b-v2-mlx-q8`. Its `convert_nemo.py` rewrites raw
    NeMo decoder keys into a different naming convention and stores
    conv weights in MLX layout. The sanitize must:

      1. Map alt-named decoder keys to the same internal namespace.
      2. Skip the conv transpose (weights are already in MLX layout).

    The two behaviors are linked: detection happens once via sentinel
    prefixes, and gates both the alt-naming branches and the
    conv-transpose skip.

    Each test below uses an MLX-format sanitize input that includes at
    least one alt-naming sentinel key to trigger the detection.
    """

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)
        # Sentinel that flags the dict as MLX-converted.
        self._sentinel = {
            "transf_decoder.token_embedding.weight": mx.zeros((1, 1)),
        }

    def _sanitize(self, weights):
        return self.model.sanitize({**self._sentinel, **weights})

    # Alt-naming cases (these would silently fall through `load_weights(
    # strict=False)` on pristine main, leaving the decoder partially
    # uninitialized).

    def test_token_embedding_no_infix(self):
        out = self._sanitize({})  # sentinel itself is the test key
        self.assertIn("decoder.embedding.weight", out)

    def test_embedding_layer_norm_no_infix(self):
        out = self._sanitize(
            {"transf_decoder.embedding_layer_norm.weight": mx.zeros((32,))}
        )
        self.assertIn("decoder.embedding_layer_norm.weight", out)

    def test_decoder_layers_alt_self_attn_names(self):
        weights = {
            "transf_decoder.layers.0.first_sub_layer.linear_q.weight": mx.zeros(
                (32, 32)
            ),
            "transf_decoder.layers.0.first_sub_layer.linear_k.weight": mx.zeros(
                (32, 32)
            ),
            "transf_decoder.layers.0.first_sub_layer.linear_v.weight": mx.zeros(
                (32, 32)
            ),
            "transf_decoder.layers.0.first_sub_layer.linear_out.weight": mx.zeros(
                (32, 32)
            ),
        }
        out = self._sanitize(weights)
        for tail in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self.assertIn(f"decoder.blocks.0.self_attn.{tail}.weight", out)

    def test_decoder_layers_alt_cross_attn_names(self):
        weights = {
            "transf_decoder.layers.1.second_sub_layer.linear_q.weight": mx.zeros(
                (32, 32)
            ),
            "transf_decoder.layers.1.second_sub_layer.linear_k.weight": mx.zeros(
                (32, 32)
            ),
            "transf_decoder.layers.1.second_sub_layer.linear_v.weight": mx.zeros(
                (32, 32)
            ),
            "transf_decoder.layers.1.second_sub_layer.linear_out.weight": mx.zeros(
                (32, 32)
            ),
        }
        out = self._sanitize(weights)
        for tail in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self.assertIn(f"decoder.blocks.1.cross_attn.{tail}.weight", out)

    def test_decoder_layers_alt_layer_norm_names(self):
        # MLX-converted layer norms use the same `layer_norm_{1,2,3}`
        # suffixes as raw NeMo, but live under the alt-naming
        # `transf_decoder.layers.<i>.` (no `_decoder.` infix) namespace.
        weights = {
            "transf_decoder.layers.2.layer_norm_1.weight": mx.zeros((32,)),
            "transf_decoder.layers.2.layer_norm_2.weight": mx.zeros((32,)),
            "transf_decoder.layers.2.layer_norm_3.weight": mx.zeros((32,)),
        }
        out = self._sanitize(weights)
        self.assertIn("decoder.blocks.2.self_attn_norm.weight", out)
        self.assertIn("decoder.blocks.2.cross_attn_norm.weight", out)
        self.assertIn("decoder.blocks.2.ff_norm.weight", out)

    def test_decoder_layers_alt_ff_names(self):
        weights = {
            "transf_decoder.layers.0.third_sub_layer.linear1.weight": mx.zeros(
                (64, 32)
            ),
            "transf_decoder.layers.0.third_sub_layer.linear2.weight": mx.zeros(
                (32, 64)
            ),
        }
        out = self._sanitize(weights)
        self.assertIn("decoder.blocks.0.ff1.weight", out)
        self.assertIn("decoder.blocks.0.ff2.weight", out)

    def test_final_layer_norm_no_infix(self):
        out = self._sanitize(
            {"transf_decoder.final_layer_norm.weight": mx.zeros((32,))}
        )
        self.assertIn("decoder.final_norm.weight", out)

    def test_head_classifier_to_output_proj(self):
        out = self._sanitize({"head.classifier.weight": mx.zeros((64, 32))})
        self.assertIn("decoder.output_proj.weight", out)

    # Conv-layout cases. With the sentinel present, conv weights must
    # NOT be transposed - they are already in MLX layout.

    def test_4d_conv_passes_through(self):
        out = self._sanitize(
            {"encoder.subsampling.conv.weight": mx.zeros((256, 3, 3, 1))}
        )
        self.assertEqual(
            out["encoder.conformer.subsampling.conv.weight"].shape, (256, 3, 3, 1)
        )

    def test_4d_conv_with_in_channels_passes_through(self):
        # MLX layout where in_channels is a typical 32, not a sentinel value.
        out = self._sanitize({"encoder.block.conv.weight": mx.zeros((64, 3, 3, 32))})
        self.assertEqual(
            out["encoder.conformer.block.conv.weight"].shape, (64, 3, 3, 32)
        )

    def test_3d_conv_passes_through(self):
        out = self._sanitize({"encoder.depthwise.conv.weight": mx.zeros((1024, 9, 1))})
        self.assertEqual(
            out["encoder.conformer.depthwise.conv.weight"].shape, (1024, 9, 1)
        )


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_model_init(self):
        self.assertIsInstance(self.model.encoder, CanaryEncoder)
        self.assertIsInstance(self.model.decoder, CanaryDecoder)

    def test_sample_rate(self):
        self.assertEqual(self.model.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
