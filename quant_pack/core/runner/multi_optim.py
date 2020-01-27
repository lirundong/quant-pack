# -*- coding: utf-8 -*-

import collections
import logging
from itertools import chain

from torch.optim.optimizer import Optimizer
from mmcv.runner import Runner, IterTimerHook, CheckpointHook, obj_from_dict

import quant_pack.core.train as training
import quant_pack.core.eval as evaluation
import quant_pack.core.wrapper as wrapper
import quant_pack.core.logger as logger
from quant_pack.core.train.qat_policies import HijackModuleOutput


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

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        super(MultiOptimRunner, self).__init__(model, batch_processor, optimizer,
                                               work_dir, log_level, logger)
        self.runtime_hook = None

    def init_optimizer(self, optimizer):
        if isinstance(optimizer, dict) and \
           all(isinstance(optim, Optimizer) for _, optim in optimizer.items()):
            return _OptimDict(optimizer)
        else:
            return super(MultiOptimRunner, self).init_optimizer(optimizer)

    def register_qat_hooks(self, loss, metrics, lr_policies, qat_policies,
                           ckpt_interval=1, runtime_hook=None):
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
        self.register_hook(CheckpointHook(interval=ckpt_interval))

        for hook in chain(metrics, qat_policies, lr_policies):
            if isinstance(hook, HijackModuleOutput):
                priority = "LOW"
            else:
                priority = "NORMAL"
            self.register_hook(hook, priority)

        if runtime_hook is not None:
            interval = runtime_hook["interval"]
            hooks = runtime_hook["hooks"]
            post_process = runtime_hook.get("post_process")
            self.inject_runtime_hooks(interval, hooks, post_process)
        else:
            self.inject_runtime_hooks(-1, [], None)

    def register_eval_hooks(self, metrics):
        for metric in metrics:
            metric_cls = metric["name"]
            metric_args = metric["args"]
            if metric_cls not in evaluation.__dict__:
                metric_cls += "Hook"
            metric_hook = evaluation.__dict__[metric_cls](**metric_args)
            self.register_hook(metric_hook)

    def register_logger_hooks(self, log_config):
        import mmcv.runner.hooks as _hooks
        log_interval = log_config["interval"]
        for info in log_config['hooks']:
            if info["type"] in logger.__dict__:
                logger_hook = obj_from_dict(
                    info, logger, default_args=dict(interval=log_interval))
            else:
                logger_hook = obj_from_dict(
                    info, _hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority="VERY_LOW")

    def inject_runtime_hooks(self, interval, hooks, post_process):
        if post_process is not None:
            runtime_hook = wrapper.WithPostprocessRuntimeHook(interval, hooks, post_process)
        else:
            runtime_hook = wrapper.RuntimeHook(interval, hooks)
        self.register_hook(runtime_hook)
        self.runtime_hook = runtime_hook

    def current_lr(self):
        if self.mode == "val" and self.optimizer is None:
            return [0., ]
        lrs = super(MultiOptimRunner, self).current_lr()
        return lrs[::-1]
