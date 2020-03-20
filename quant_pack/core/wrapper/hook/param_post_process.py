# -*- coding: utf-8 -*-

import copy


class ParamPassThrough:

    plot_method = "naive_plot_param"

    def __init__(self, apply_to):
        self.apply_to = apply_to

    def after_iter(self, input_reg, outputs):
        return copy.copy(input_reg[self.apply_to])
