# -*- coding: utf-8 -*-

from torchvision.models import MobileNetV2
from torchvision.models.utils import load_state_dict_from_url

from .idq import IDQ

__all__ = ["IDQMobileNetV2", "mobilenet_v2_idq"]

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class IDQMobileNetV2(MobileNetV2, IDQ):

    def __init__(self, num_classes=1000, width_mult=1.0, kw=4, ka=4, fp_layers=None, align_zero=True):
        MobileNetV2.__init__(self, num_classes, width_mult)
        IDQ.__init__(self, MobileNetV2.forward, kw, ka, fp_layers, align_zero)


def mobilenet_v2_idq(pretrained=False, progress=True, **kwargs):
    model = IDQMobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        model.load_state_dict(state_dict)
    return model
