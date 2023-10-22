import json
from unicodedata import bidirectional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
import sys
from CONSTANTS import DATA_DIR
import transformers
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutput,
)
import os
import math
from collections.abc import Sequence

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

class BottleneckBlock(nn.Module):
    def __init__(self, args, input_dim, int_dim, output_dim, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, int_dim, (1, 1), stride=stride)
        self.conv2 = nn.Conv2d(
            int_dim, int_dim, (3, 3), padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(int_dim, output_dim, (1, 1))

        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.bn1 = torch.nn.BatchNorm2d(input_dim)
        self.bn2 = torch.nn.BatchNorm2d(int_dim)
        self.bn3 = torch.nn.BatchNorm2d(int_dim)

        if input_dim != output_dim:
            self.proj_shortcut = nn.Conv2d(
                input_dim, output_dim, (1, 1), stride=stride)
        else:
            self.proj_shortcut = None

    def init(self):
        xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, features):
        x = self.bn1(features)
        x = F.relu(x)
        if self.proj_shortcut:
            features = self.proj_shortcut(features)
        x = self.feature_map_drop(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv3(x)

        return x + features

def load_embedding(args, bert_model, embed, pool):
    if embed == 'flow':
        file_path = os.path.join(DATA_DIR[args.dataset], 'embeddings', f'{os.path.basename(bert_model)}_{pool}_flow.pt')
    else:
        file_path = os.path.join(DATA_DIR[args.dataset], 'embeddings', f'{os.path.basename(bert_model)}_{pool}.pt')
    embedding = torch.load(file_path)
    return embedding

class TailEmbedding(nn.Module):
    def __init__(self, args):
        super(TailEmbedding, self).__init__()
        embedding = load_embedding(args, args.tail_bert_model, args.tail_embed, args.tail_bert_pool)
        self.embedding_dim = embedding.shape[1]

        if args.tail_embed == 'normalize':
            embedding -= torch.mean(embedding, dim=0)
            embedding = F.normalize(embedding, p=2, dim=1)
            self.tail_embedding = nn.Embedding.from_pretrained(embedding)
        elif 'mlp' in args.tail_embed:
            embedding = F.normalize(embedding, p=2, dim=1)
            self.tail_embedding = nn.Embedding.from_pretrained(embedding)
            self.tail_mlp = nn.Sequential(*[nn.Linear(self.embedding_dim, self.embedding_dim), 
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(self.embedding_dim, self.embedding_dim)])
            if 'res' in args.tail_embed:
                torch.nn.init.zeros_(self.tail_mlp[-1].weight)
        elif args.tail_embed == 'flow':
            embedding = F.normalize(embedding, p=2, dim=1)
            self.tail_embedding = nn.Embedding.from_pretrained(embedding)
            self.embedding_dim = embedding.shape[1]
        elif args.tail_embed == 'default':
            self.tail_embedding = nn.Embedding.from_pretrained(embedding)
        else: 
            raise NotImplementedError
        self.args = args
    def get_transpose(self):
        return self.get_weight().t()
    def get_weight(self):
        if self.args.tail_embed in ['default', 'normalize', 'flow']:
            return self.tail_embedding.weight
        elif self.args.tail_embed == 'mlp':
            embedding = self.tail_mlp(self.tail_embedding.weight)
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        elif self.args.tail_embed == 'res_mlp':
            embedding = self.tail_mlp(self.tail_embedding.weight)+self.tail_embedding.weight
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        else:
            raise NotImplementedError


class HeadEmbedding(nn.Module):
    def __init__(self, args):
        super(HeadEmbedding, self).__init__()
        if args.head_bert_pool in ['prior', 'cls', 'mean']:
            embedding = load_embedding(args, args.head_bert_model, '', args.head_bert_pool)
            args.embedding_dim = embedding.shape[1]
            self.embedding_dim = embedding.shape[1]
            self.ent_embedding = nn.Embedding.from_pretrained(embedding)
        elif args.head_bert_pool == 'prompt':
            self.init_bert(args)
        else:
            raise NotImplementedError
        self.args = args

    def init_bert(self, args):
        self.bert = transformers.AutoModel.from_pretrained(args.head_bert_model)
        if not args.unfreeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        args.embedding_dim = self.bert.config.hidden_size

        if args.prefix_dim == 0:
            args.prefix_dim = self.bert.config.hidden_size

        self.num_prefix_layers = self.bert.config.num_hidden_layers
        if args.num_prefixes > 0:
            self.prefix_embedding = torch.nn.Embedding(self.num_prefix_layers, args.prefix_dim*args.num_prefixes)
        self.total_num_prefixes = args.num_prefixes

        if args.prefix_embed == 'mlp':
            self.prefix_embed_transform = nn.Sequential(
                            nn.Dropout(p=0.1),
                            nn.Linear(args.prefix_dim, self.bert.config.hidden_size//2),
                            nn.ReLU(),
                            nn.Dropout(p=0.1),
                            nn.Linear(self.bert.config.hidden_size//2, self.bert.config.hidden_size),)
        else:
            raise NotImplementedError
        
        self.prefix_embed_ln = nn.LayerNorm(self.bert.config.hidden_size)

        if args.layer_aggr == 'lin_comb':
            self.prefix_lin_comb = nn.Parameter(torch.zeros(
                    self.bert.config.num_hidden_layers+1, dtype=torch.float32, requires_grad=True))
            
            self.softmax = torch.nn.Softmax(dim=0)

            if args.use_prefix_projection:
                self.prefix_proj = nn.Linear(args.embedding_dim, args.embedding_dim)
            self.embedding_dim = self.bert.config.hidden_size
        else:
            raise NotImplementedError

        self.embedding_mask = nn.Parameter(torch.zeros(self.total_num_prefixes), requires_grad=False)
    
    def prefix_encoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        rel_id=None,
        **kwargs
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        embedding_mask = self.embedding_mask.expand((attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], -1))
        attention_mask = torch.cat((embedding_mask, attention_mask), dim=3)

        next_decoder_cache = () if use_cache else None
        # print(rel_embedded.shape)
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if self.total_num_prefixes > 0:
                prefix_embs = []
                if self.args.num_prefixes > 0:
                    prefix_emb = self.prefix_embedding(torch.tensor(i, device=self.bert.device)).view(-1, self.args.num_prefixes, self.args.prefix_dim).expand(rel_id.shape[0], -1, -1)
                    # print(prefix_emb.shape)
                    prefix_embs.append(prefix_emb)
                assert len(prefix_embs) > 0
                prefix_emb = torch.cat(prefix_embs, dim=1)
                if self.args.prefix_embed == 'mlp':
                    prefix_emb = self.prefix_embed_transform(prefix_emb)
                else:
                    raise NotImplementedError
                prefix_emb = self.prefix_embed_ln(prefix_emb)
                if i==0:
                    hidden_states = torch.cat([prefix_emb, hidden_states], dim=1)
                    all_hidden_states = all_hidden_states + (hidden_states,)
                else:
                    hidden_states = hidden_states[:, self.total_num_prefixes:, :]
                    hidden_states = torch.cat([prefix_emb, hidden_states], dim=1)
            elif i==0 and self.total_num_prefixes == 0:
                all_hidden_states = all_hidden_states + (hidden_states,)


            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.bert.encoder.config, "gradient_checkpointing", False) and self.bert.encoder.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def prefix_bert_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rel_id=None,
        **kwargs
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict

        if self.bert.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.bert.config.use_cache
            use_cache = False
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        encoder_outputs = self.prefix_encoder_forward(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rel_id=rel_id,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
            

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def probe_bert(self, encoded):
        encoded['output_hidden_states'] = True

        if self.total_num_prefixes > 0:
            bert_embs = torch.stack([emb for emb in self.prefix_bert_forward(**encoded).hidden_states], dim=1)
        else:
            encoded.pop('ent_id')
            encoded.pop('rel_id')
            bert_embs = torch.stack([emb for emb in self.bert(**encoded).hidden_states], dim=1)
        attn_mask = encoded['attention_mask'].view((encoded['attention_mask'].shape[0], 1, encoded['attention_mask'].shape[1], 1))

        if self.args.use_prefix_projection:
            proj_embs = self.prefix_proj(bert_embs)
        else:
            proj_embs = bert_embs
        
        if self.args.span_extraction == 'max':
            proj_embs[:, :, self.total_num_prefixes:, :][torch.logical_not(attn_mask).expand_as(proj_embs[:, :, self.total_num_prefixes:, :])] = float('-inf')
            pooled_embs = torch.max(proj_embs, dim=2).values
        elif self.args.span_extraction == 'mean':
            proj_embs[:, :, self.total_num_prefixes:, :] = proj_embs[:, :, self.total_num_prefixes:, :]*attn_mask
            
            pooled_embs = torch.sum(proj_embs, dim=2)
            denominators = torch.sum(encoded['attention_mask'], dim=1, keepdim=True) + self.total_num_prefixes
            pooled_embs = pooled_embs/denominators.unsqueeze(-1)
        else:
            raise NotImplementedError

        if self.args.layer_aggr == 'lin_comb':    
            scalars = self.softmax(self.prefix_lin_comb).unsqueeze(-1).unsqueeze(0)
            
            ent_embedded = torch.sum(pooled_embs*scalars, dim=1)
        else:
            raise NotImplementedError

        return ent_embedded


    def get_weight(self):
        if self.args.head_bert_pool in ['prior', 'mean', 'cls']:
            return self.ent_embedding.weight
        elif self.args.head_bert_pool == 'prompt':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, encoded):
        if self.args.head_bert_pool in ['prior', 'mean', 'cls']:
            return self.ent_embedding(encoded['ent_id'])
        elif self.args.head_bert_pool == 'prompt':
            return self.probe_bert(encoded)
        else:
            raise NotImplementedError
        
        

class PretrainedBertResNet(nn.Module):
    def __init__(self, args):
        super(PretrainedBertResNet, self).__init__()

        self.head_embedding = HeadEmbedding(args)
        self.tail_embedding = TailEmbedding(args)

        self.rel_embedding = torch.nn.Embedding(
            args.num_relations, self.head_embedding.embedding_dim)

        self.reshape_len = args.reshape_len

        input_channels = 2

        self.conv1 = nn.Conv1d(input_channels, self.reshape_len**2, kernel_size=1)

        bottlenecks = []

        input_dim = self.head_embedding.embedding_dim
        output_dim = self.head_embedding.embedding_dim
        for i in range(args.resnet_num_blocks):
            bottlenecks.append(BottleneckBlock(
                args, input_dim, output_dim//4, output_dim, stride=1))
            input_dim = output_dim
            bottlenecks.extend([BottleneckBlock(
                args, input_dim, output_dim//4, output_dim) for _ in range(min(args.resnet_block_depth, 2)-1)])
            output_dim *= 2
        self.output_dim = output_dim//2
        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.bias = nn.Parameter(torch.zeros((1), dtype=torch.float), requires_grad=True)

        self.fc = nn.Linear(self.output_dim, self.tail_embedding.embedding_dim)

        self.bn0 = torch.nn.BatchNorm1d(input_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.output_dim)
        self.args = args
        self.init()
        self.prelu = torch.nn.PReLU(self.tail_embedding.embedding_dim)

    def re_init_head(self, args):
        self.args = args
        self.head_embedding = HeadEmbedding(self.args)

    def init(self):
        xavier_normal_(self.rel_embedding.weight)

    def query_emb(self, encoded):
        batch_size = encoded['ent_id'].shape[0]

        rel_embedded = self.rel_embedding(encoded['rel_id']).unsqueeze(1)
        ent_embedded = self.head_embedding(encoded).unsqueeze(1)        

        stacked_inputs = torch.cat(
            [ent_embedded, rel_embedded], 1)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2).view(
            batch_size, self.head_embedding.embedding_dim, self.reshape_len, self.reshape_len).contiguous()

        x = self.bottlenecks(x)

        x = self.bn1(x)
        x = F.relu(x)

        x = torch.mean(x.view(batch_size, self.output_dim, -1), dim=2)
        x = self.hidden_drop(x)
        x = self.fc(x)

        x = self.prelu(x)
        x = self.hidden_drop(x)

        return x

    def forward(self, encoded):

        tail_embeddings = self.tail_embedding.get_weight()
        x = self.query_emb(encoded)
        # return x

        
        scores = torch.mm(x, tail_embeddings.t())
        scores += self.bias
        return scores

