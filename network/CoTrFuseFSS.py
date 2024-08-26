# coding=utf-8
# double encoder CNN decoder

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from segmentation_models_pytorch.encoders import get_encoder

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

logger = logging.getLogger(__name__)
import torch.nn.functional as F
import torch
# import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from segmentation_models_pytorch.unet.decoder import DecoderBlock, CenterBlock
from segmentation_models_pytorch.base import SegmentationHead
from .STCF import SCF_block, DAS_block
from functools import partial

class CosineKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', norm='L2'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm.upper()

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        if self.norm == "L2":
            loss = ((inputs - targets)**2).mean(dim=1)
        else:
            loss = (inputs - targets).mean(dim=1)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets / self.alpha, dim=1)

        loss = -(outputs * labels).mean(dim=1) * (self.alpha ** 2)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


# MiB Losses
class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class CosineLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.crit = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = 1 - self.crit(x, y)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = loss
        return - loss
    
class CoTrFuseModified(nn.Module):
    def __init__(self, num_old_classes, num_new_classes, config, img_size=224, model_name="resnet50"):
        super().__init__()
        # ... (encoder CNN, encoder Transformer, modulo di fusione da CoTrFuse originale)
        self.old_classifier = nn.Linear(...)  # Classificatore per le vecchie classi
        self.new_classifier = nn.Linear(...)  # Classificatore per le nuove classi
        self.prototype_memory = {}  # Memoria per i prototipi (dizionario)
        # ... (resto dell'inizializzazione da SwinUnet)

    def forward(self, x, old_classes=True, new_classes=True):
        # ... (estrazione caratteristiche con encoder da SwinUnet)
        if old_classes:
            old_logits = self.old_classifier(...)
        if new_classes:
            new_logits = self.new_classifier(...)
            # Calcolo prototipi dinamici (adattamento necessario)
            prototypes = self._calculate_prototypes(new_logits, new_targets)  # Assumendo new_targets disponibili
            self.prototype_memory.update(prototypes)  # Aggiorna la memoria dei prototipi
        # ... (combinazione di old_logits e new_logits se necessario)
        return old_logits, new_logits 