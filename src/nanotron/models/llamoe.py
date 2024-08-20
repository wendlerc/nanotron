"""PyTorch GPT-3 MoE model."""

from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron.config import LlamaConfig, LlaMoEConfig, ParallelismArgs
from nanotron.models import llama
from nanotron.models.llama import CausalSelfAttention, LlamaDecoderLayer, LlamaForTraining, LlamaModel
from nanotron.models.moe import (
    dMoE,
)
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear
from src.nanotron.random import RandomStates


@contextmanager
def replace_moe_decoder(llamaconfig: LlaMoEConfig):
    orig = llama.PipelineBlock
    try:

        def create_pp_block(module_builder, module_kwargs, **kwargs):
            if module_builder is LlamaDecoderLayer:
                # llama's GPT module is trying to instantiate a llama GPTBlock.
                # Let's return a PipelineBlock with a LlamaDecoderLayer instead.
                # This also requires to replace starcoders2's config with llama's config.
                module_kwargs["config"] = llamaconfig
                return orig(module_builder=LlaMoeBlock, module_kwargs=module_kwargs, **kwargs)
            # Else, they are setting up other modules, which we also want unchanged.
            return orig(module_builder=module_builder, module_kwargs=module_kwargs, **kwargs)

        llama.PipelineBlock = create_pp_block
        yield
    finally:
        llama.PipelineBlock = orig


@contextmanager
def replace_llama_moe_model(LlaMoEConfig: LlaMoEConfig):
    orig = llama.LlamaModel
    try:

        def create_moe_model(
            config: LlamaConfig,
            parallel_context: ParallelContext,
            parallel_config: Optional[ParallelismArgs],
        ):
            return LlaMoEModel(LlaMoEConfig, parallel_context, parallel_config)

        llama.LlamaModel = create_moe_model
        yield
    finally:
        llama.LlamaModel = orig


class LlaMoeBlock(nn.Module):
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super(LlaMoeBlock, self).__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config, parallel_config=parallel_config, tp_pg=tp_pg, layer_idx=layer_idx
        )

        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = dMoE(
            config=config,
            parallel_config=parallel_config,
            parallel_context=parallel_context,
            use_glu=True,
        )
        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: torch.Tensor | TensorPointer,
        sequence_mask: torch.Tensor | TensorPointer,
        aux_losses: Dict[str, Union[torch.Tensor, TensorPointer]],
    ) -> Dict[str, torch.Tensor | TensorPointer]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states=hidden_states)
        hidden_states = mlp_output["hidden_states"]

        for key, value in mlp_output.items():
            if key != "hidden_states":
                aux_losses[key] = aux_losses[key] + value

        hidden_states = hidden_states + residual

        return hidden_states, output["sequence_mask"], aux_losses

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask: torch.Tensor,
        aux_losses: Dict[str, Union[torch.Tensor, TensorPointer]],
    ) -> List[torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, sequence_mask, aux_losses)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
        aux_losses: Dict[str, Union[torch.Tensor, TensorPointer]],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:

        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, sequence_mask, aux_losses = self._checkpointed_forward(
                hidden_states, sequence_mask, aux_losses
            )
        else:
            hidden_states, sequence_mask, aux_losses = self._core_forward(hidden_states, sequence_mask, aux_losses)

        return {"hidden_states": hidden_states, "sequence_mask": sequence_mask, "aux_losses": aux_losses}


class LlaMoEModel(LlamaModel):
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        with replace_moe_decoder(config):
            super().__init__(config.as_llama(), parallel_context, parallel_config)

        # need to adapt the decoder list because we pass the aux_losses around
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlaMoeBlock,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "parallel_context": parallel_context,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "sequence_mask", "aux_losses"},
                    module_output_keys={"hidden_states", "sequence_mask", "aux_losses"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
        input_mask: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
        aux_losses: Dict[str, Union[torch.Tensor, TensorPointer]],
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

        hidden_encoder_states = {
            "hidden_states": output["input_embeds"],
            "sequence_mask": input_mask,
            "aux_losses": aux_losses,
        }
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)
        # return hidden_encoder_states["hidden_states"]

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return {"sharded_logits": fp32_sharded_logits, "aux_losses": hidden_encoder_states["aux_losses"]}


class LlaMoEForTraining(LlamaForTraining):
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        with replace_llama_moe_model(config):
            super().__init__(config.as_llama(), parallel_context, parallel_config)
        self.config = config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # aux_losses are used for load balancing in case of MoEs
        aux_losses = {
            "load_balancing_loss": (
                torch.zeros(1, device=input_ids.device)
                if not isinstance(input_ids, TensorPointer)
                else TensorPointer(self.input_pp_rank)
            ),
            "z_loss": (
                torch.zeros(1, device=input_ids.device)
                if not isinstance(input_ids, TensorPointer)
                else TensorPointer(self.input_pp_rank)
            ),
        }
        model_output = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            aux_losses=aux_losses,
        )
        outputs = self.loss(
            sharded_logits=model_output["sharded_logits"],
            label_ids=label_ids,
            label_mask=label_mask,
        )

        if isinstance(model_output["aux_losses"], dict):
            for key, value in model_output["aux_losses"].items():
                outputs[key] = value
        return outputs

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = (
            model_config.intermediate_size
            if model_config.intermediate_size is not None
            else 4 * model_config.hidden_size
        )
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        # active experts + routing
        mlp_cost = (
            2 * d_ff * model_config.hidden_size * model_config.num_experts_per_tok
            + model_config.hidden_size * model_config.moe_num_experts
        )
        att_cost = 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            LlaMoeBlock: att_cost + mlp_cost,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads
        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
            num_experts=self.config.moe_num_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
        )
        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
    num_experts=8,
    num_experts_per_tok=2,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    ## v logits
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    ## 2nd layer
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size
    # moe routing
    if num_experts > 1:
        decoder_ffn_router_flops_fwd = 2 * num_layers * batch_size * seq_len * (hidden_size) * num_experts
        decoder_ffn_1_flops_fwd *= num_experts_per_tok
        decoder_ffn_2_flops_fwd *= num_experts_per_tok
    else:
        decoder_ffn_router_flops_fwd = 0

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
        + decoder_ffn_router_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    return model_flops, hardware_flops
