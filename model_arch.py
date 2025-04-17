"""
MobileLlamaForCausalLM(
  (model): MobileLlamaModel(
    (embed_tokens): Embedding(32000, 2048, padding_idx=0)
    (layers): ModuleList(
      (0-23): 24 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
    (vision_tower): CLIPVisionTower(
      (vision_tower): CLIPVisionModel(
        (vision_model): CLIPVisionTransformer(
          (embeddings): CLIPVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
            (position_embedding): Embedding(577, 1024)
          )
          (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-23): 24 x CLIPEncoderLayer(
                (self_attn): CLIPAttention(
                  (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                  (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (mm_projector): LDPNetProjector(
      (model): LDPBlock(
        (mlp): Sequential(
          (0): Identity()
          (1): Linear(in_features=1024, out_features=2048, bias=True)
          (2): GELU(approximate='none')
          (3): Linear(in_features=2048, out_features=2048, bias=True)
        )
        (mb_block): Sequential(
          (0): Identity()
          (1): InvertedResidual(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=False)
                (1): LayerNormAct2d(
                  (2048,), eps=1e-05, elementwise_affine=True
                  (drop): Identity()
                  (act): Identity()
                )
                (2): Hardswish()
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
              )
              (2): Conv2dNormActivation(
                (0): Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): LayerNormAct2d(
                  (2048,), eps=1e-05, elementwise_affine=True
                  (drop): Identity()
                  (act): Identity()
                )
              )
            )
          )
          (2): InvertedResidual(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=2048, bias=False)
                (1): LayerNormAct2d(
                  (2048,), eps=1e-05, elementwise_affine=True
                  (drop): Identity()
                  (act): Identity()
                )
                (2): Hardswish()
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
              )
              (2): Conv2dNormActivation(
                (0): Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): LayerNormAct2d(
                  (2048,), eps=1e-05, elementwise_affine=True
                  (drop): Identity()
                  (act): Identity()
                )
              )
            )
          )
        )
      )
    )
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
"""