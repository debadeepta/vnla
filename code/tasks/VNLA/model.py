# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import padding_idx
from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent

class EncoderLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                       dropout_ratio, device, bidirectional=False, num_layers=1):

        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers // (2 if bidirectional else 1)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions)
        self.device = device

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(
            (self.num_layers * self.num_directions, batch_size, self.hidden_size),
            dtype=torch.float,
            device=self.device)

        c0 = torch.zeros(
            (self.num_layers * self.num_directions, batch_size, self.hidden_size),
            dtype=torch.float,
            device=self.device)

        return h0, c0

    def forward(self, inputs, lengths):
        # Sort inputs by length
        sorted_lengths, forward_index_map = lengths.sort(0, True)

        inputs = inputs[forward_index_map]
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        state = self.init_state(inputs)

        packed_embeds = pack_padded_sequence(embeds, sorted_lengths.to('cpu'), batch_first=True)
        enc_h, state = self.lstm(packed_embeds, state)

        state = (self.encoder2decoder(state[0]), self.encoder2decoder(state[1]))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)

        # Unsort outputs
        _, backward_index_map = forward_index_map.sort(0, False)
        ctx = ctx[backward_index_map]

        return ctx, state


class Attention(nn.Module):

    def __init__(self, dim, coverage_dim=None):
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        if coverage_dim is not None:
            self.cov_rnn = nn.GRU(dim * 2 + 1, coverage_dim, 1)
            self.cov_linear = nn.Linear(coverage_dim, dim)

    def forward(self, h, context, mask=None, cov=None):
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        if cov is not None:
            context = context + self.cov_linear(cov)

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        # Update coverage vector
        if hasattr(self, 'cov_rnn') and hasattr(self, 'cov_linear'):
            cov_expand = cov.view(-1, cov.size(2))
            context_expand = context.view(-1, context.size(2))
            h_expand = h.unsqueeze(1).expand(-1, cov.size(1), -1).contiguous().view(-1, h.size(1))
            attn_expand = attn.unsqueeze(2).view(-1, 1)
            concat_input = torch.cat((context_expand, h_expand, attn_expand), 1)
            new_cov, _ = self.cov_rnn(concat_input.unsqueeze(0), cov_expand.unsqueeze(0))
            new_cov = new_cov.squeeze(0).view_as(cov)
        else:
            new_cov = None

        return h_tilde, attn, new_cov


class AskAttnDecoderLSTM(nn.Module):

    def __init__(self, hparams, agent_class, device):

        super(AskAttnDecoderLSTM, self).__init__()

        self.device = device

        self.nav_embedding = nn.Embedding(agent_class.n_input_nav_actions(),
            hparams.nav_embed_size, padding_idx=padding_idx)
        self.ask_embedding = nn.Embedding(agent_class.n_input_ask_actions(hparams),
            hparams.ask_embed_size)

        lstm_input_size = hparams.nav_embed_size + hparams.ask_embed_size + hparams.img_feature_size

        self.budget_embedding = nn.Embedding(hparams.max_ask_budget, hparams.budget_embed_size)

        self.drop = nn.Dropout(p=hparams.dropout_ratio)

        self.lstm = nn.LSTM(
            lstm_input_size, hparams.hidden_size, hparams.num_lstm_layers,
            dropout=hparams.dropout_ratio if hparams.num_lstm_layers > 1 else 0,
            bidirectional=False)

        self.attention_layer = Attention(hparams.hidden_size,
            coverage_dim=hparams.coverage_size
            if hasattr(hparams, 'coverage_size') else None)

        self.nav_predictor = nn.Linear(hparams.hidden_size, agent_class.n_output_nav_actions())

        ask_predictor_input_size = hparams.hidden_size * 2 + \
            agent_class.n_output_nav_actions() + hparams.img_feature_size + \
            hparams.budget_embed_size

        ask_predictor_layers = []
        current_layer_size = ask_predictor_input_size
        next_layer_size = hparams.hidden_size

        if not hasattr(hparams, 'num_ask_layers'):
            hparams.num_ask_layers = 1

        for i in range(hparams.num_ask_layers):
            ask_predictor_layers.append(nn.Linear(current_layer_size, next_layer_size))
            ask_predictor_layers.append(nn.ReLU())
            ask_predictor_layers.append(nn.Dropout(p=hparams.dropout_ratio))
            current_layer_size = next_layer_size
            next_layer_size //= 2
        ask_predictor_layers.append(nn.Linear(current_layer_size, agent_class.n_output_ask_actions(hparams)))

        self.ask_predictor = nn.Sequential(*tuple(ask_predictor_layers))

        self.backprop_softmax = hparams.backprop_softmax
        self.backprop_ask_features = hparams.backprop_ask_features

    def _lstm_and_attend(self, nav_action, ask_action, feature, h, ctx, ctx_mask,
        budget=None, cov=None):

        nav_embeds = self.nav_embedding(nav_action)
        ask_embeds = self.ask_embedding(ask_action)

        lstm_inputs = [nav_embeds, ask_embeds, feature]

        concat_lstm_input = torch.cat(lstm_inputs, dim=1)
        drop = self.drop(concat_lstm_input)
        output, new_h = self.lstm(drop.unsqueeze(0), h)

        output = output.squeeze(0)

        output_drop = self.drop(output)

        # Attention
        h_tilde, alpha, new_cov = self.attention_layer(output_drop, ctx, ctx_mask, cov=cov)

        return h_tilde, alpha, output_drop, new_h, new_cov

    def forward(self, nav_action, ask_action, feature, h, ctx, ctx_mask,
                nav_logit_mask, ask_logit_mask,
                budget=None, cov=None):

        h_tilde, alpha, output_drop, new_h, new_cov = self._lstm_and_attend(
            nav_action, ask_action, feature, h, ctx, ctx_mask, budget=budget, cov=cov)

        # Predict nav action.
        nav_logit = self.nav_predictor(h_tilde)
        nav_logit.data.masked_fill_(nav_logit_mask, -float('inf'))
        nav_softmax = F.softmax(nav_logit, dim=1)
        if not self.backprop_softmax:
            nav_softmax = nav_softmax.detach()

        assert budget is not None
        budget_embeds = self.budget_embedding(budget)
        ask_predictor_inputs = [h_tilde, output_drop, feature, nav_softmax, budget_embeds]

        # Predict ask action.
        concat_ask_predictor_input = torch.cat(ask_predictor_inputs, dim=1)

        if not self.backprop_ask_features:
            concat_ask_predictor_input = concat_ask_predictor_input.detach()

        ask_logit = self.ask_predictor(concat_ask_predictor_input)
        ask_logit.data.masked_fill_(ask_logit_mask, -float('inf'))

        return new_h, alpha, nav_logit, nav_softmax, ask_logit, new_cov

    def forward_nav(self, nav_action, ask_action, feature, h, ctx, ctx_mask,
                    nav_logit_mask, budget=None, cov=None):

        h_tilde, alpha, output_drop, new_h, new_cov = self._lstm_and_attend(
            nav_action, ask_action, feature, h, ctx, ctx_mask, budget=budget, cov=cov)

        # Predict nav action.
        nav_logit = self.nav_predictor(h_tilde)
        nav_logit.data.masked_fill_(nav_logit_mask, -float('inf'))
        nav_softmax = F.softmax(nav_logit, dim=1)

        return new_h, alpha, nav_logit, nav_softmax, new_cov


class AttentionSeq2SeqModel(nn.Module):

    def __init__(self, vocab_size, hparams, device):
        super(AttentionSeq2SeqModel, self).__init__()
        enc_hidden_size = hparams.hidden_size // 2 \
            if hparams.bidirectional else hparams.hidden_size
        self.encoder = EncoderLSTM(vocab_size,
                              hparams.word_embed_size,
                              enc_hidden_size,
                              padding_idx,
                              hparams.dropout_ratio,
                              device,
                              bidirectional=hparams.bidirectional,
                              num_layers=hparams.num_lstm_layers)

        if 'verbal' in hparams.advisor:
            agent_class = VerbalAskAgent
        elif hparams.advisor == 'direct':
            agent_class = AskAgent
        else:
            sys.exit('%s advisor not supported' % hparams.advisor)
        self.decoder = AskAttnDecoderLSTM(hparams, agent_class, device)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def decode_nav(self, *args, **kwargs):
        return self.decoder.forward_nav(*args, **kwargs)


