# -*- coding: utf-8 -*-

from tempfile import NamedTemporaryFile

import torch
import onnx


def track_bn_folding_mapping(model, dummy_input):
    bn_conv_mappings = []
    with NamedTemporaryFile("wb") as f:
        torch.onnx.export(model, dummy_input, f)
        onnx_model = onnx.load_model(f.name)
        for i in range(len(onnx_model.graph.node) - 1, -1, -1):
            node = onnx_model.graph.node[i]
            if node.op_type == "BatchNormalization":
                bn_layer_name = node.input[1].replace(".weight", "")
                if i - 1 >= 0 and onnx_model.graph.node[i - 1].op_type == "Conv":
                    conv_layer_name = onnx_model.graph.node[i - 1].input[1].replace(".weight", "")
                    bn_conv_mappings.append((bn_layer_name, conv_layer_name))
    return bn_conv_mappings


def build_runtime_hooks(cfg):
    raise NotImplementedError()
