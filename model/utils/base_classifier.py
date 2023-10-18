import torch
from torch import nn
from transformers import BartConfig, BertConfig
from transformers.activations import ACT2FN


class TransformLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(TransformLayer, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[activation_function]
        if layer_norm_eps is None:
            self.LayerNorm = nn.LayerNorm(hidden_size)
        else:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            vocab_size,
    ):
        super(DecoderLayer, self).__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states) -> torch.Tensor:
        return self.decoder(hidden_states)


class BaseClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(BaseClassifier, self).__init__()

        self.vocab_size = vocab_size

        self.transform_layer = TransformLayer(
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder_layer = DecoderLayer(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    @classmethod
    def create(cls, vocab_size, hidden_size, activation_function, layer_norm_eps=None):
        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, last_hidden_states):
        hidden_states = self.transform_layer(last_hidden_states)
        prediction = self.decoder_layer(hidden_states)
        return prediction


class BartClassifier(BaseClassifier):

    @classmethod
    def create(
        cls,
        config: BartConfig,
        vocab_size,
        **kwargs
    ):
        return super().create(
            vocab_size=vocab_size,
            hidden_size=config.d_model,
            activation_function=config.activation_function,
        )


class BertClassifier(BaseClassifier):
    @classmethod
    def create(
            cls,
            config: BertConfig,
            vocab_size,
            **kwargs
    ):
        return super().create(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            activation_function=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
        )
