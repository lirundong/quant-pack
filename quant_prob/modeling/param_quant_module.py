# -*- coding: utf-8 -*-

import torch

if torch.cuda.is_available():
    from quant_prob.modeling.quantizers.cuda_param_linear_quantizer import cuda_fake_linear_quant
    quantizer = cuda_fake_linear_quant
else:
    from quant_prob.modeling.quantizers.param_linear_quantizer import fake_linear_quant
    quantizer = fake_linear_quant

