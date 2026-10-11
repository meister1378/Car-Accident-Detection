"""Inter-stage intervention for the existing three-stage LTContext implementation.

Inputs use B x 3072 x T features and a B x 1 x T validity mask. Outputs use
stage x B x 9 x T raw logits: acceleration [0:4], steering [4:7], reserved [7:9].
Taskwise values sum to two plus the reserved joint probability mass, so this
intervention changes group amplitude as well as direct cross-group derivatives.
Reserved values are preserved for identical logits, not across diverged models.
The mode is not in state_dict: every loader must restore it from metadata.
"""
from types import MethodType

import torch

MODES = ('joint9', 'taskwise_keep_aux')


def intermediate_probabilities(logits, mode):
    if logits.ndim != 3 or logits.shape[1] != 9:
        raise ValueError('Expected B x 9 x T LTC logits')
    joint = logits.softmax(dim=1)
    if mode == 'joint9':
        return joint
    if mode == 'taskwise_keep_aux':
        # Channels 7:9 have no direct classification target in this experiment.
        # Keep their legacy values so we do not introduce a third normalization
        # rule, remove channels, or introduce auxiliary supervision in this A/B.
        return torch.cat((logits[:, :4].softmax(dim=1),
                          logits[:, 4:7].softmax(dim=1), joint[:, 7:9]), dim=1)
    raise ValueError(mode)


def _forward(self, inputs, masks):
    out, feature = self.stage1(inputs, masks)
    output_list = [out]
    feature = self.dim_reduction(feature)
    for stage in self.stages:
        out, feature = stage(
            intermediate_probabilities(out, self.intermediate_normalization) * masks,
            prev_stage_feat=feature * masks, masks=masks)
        output_list.append(out)
    return torch.stack(output_list, dim=0)


def set_intermediate_normalization(model, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if not all(hasattr(model, attr) for attr in ('stage1', 'stages', 'dim_reduction')):
        raise TypeError('Expected the existing LTContext model')
    model.intermediate_normalization = mode
    model.forward = MethodType(_forward, model)
    return model
