# -*- coding: utf-8 -*-

import collections
from itertools import chain

from torch.optim.optimizer import Optimizer
from mmcv.runner import Runner, IterTimerHook

import quant_pack.core.train as training
import quant_pack.core.eval as evaluation


class _OptimDict(dict):

    def state_dict(self):
        ret = collections.OrderedDict()
        for name, optim in self.items():
            ret[name] = optim.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        for name, states in state_dict.items():
            self[name].load_state_dict(states)

    @property
    def param_groups(self):
        ret = []
        for _, optim in self.items():
            ret += optim.param_groups
        return ret


class MultiOptimRunner(Runner):

    def init_optimizer(self, optimizer):
        if isinstance(optimizer, dict) and \
           all(isinstance(optim, Optimizer) for _, optim in optimizer.items()):
            return _OptimDict(optimizer)
        else:
            return super(MultiOptimRunner, self).init_optimizer(optimizer)

    def register_qat_hooks(self, loss, metrics, lr_policies, qat_policies):
        assert isinstance(loss, dict)
        assert isinstance(metrics, (tuple, list))
        assert isinstance(lr_policies, (tuple, list))
        assert isinstance(qat_policies, (tuple, list))

        loss = training.build_loss(loss)
        metrics = training.build_metrics(*metrics)
        lr_policies = training.build_lr_policies(*lr_policies)
        qat_policies = training.build_qat_policies(*qat_policies)

        # make sure loss firstly getting ready after `batch_processor`
        self.register_hook(loss, priority="HIGH")
        self.register_hook(IterTimerHook())

        for hook in chain(metrics, qat_policies, lr_policies):
            self.register_hook(hook)

    def register_eval_hooks(self, metrics):
        for metric in metrics:
            metric_cls = metric["name"]
            metric_args = metric["args"]
            if metric_cls not in evaluation.__dict__:
                metric_cls += "Hook"
            metric_hook = evaluation.__dict__[metric_cls](**metric_args)
            self.register_hook(metric_hook)

    def inject_runtime_hooks(self, runtime_hooks):
        raise NotImplementedError()

    def current_lr(self):
        lrs = super(MultiOptimRunner, self).current_lr()
        return lrs[::-1]
