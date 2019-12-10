# -*- coding: utf-8 -*-

from . import diagnoser as _diagnoser
from ._registry import get_tasks
from .grad_conflict import get_grad_analysis

__all__ = ["get_diagnoser", "get_tasks", "get_grad_analysis"]


def get_diagnoser(diagnoser_type, module, *diagnoser_args, **diagnoser_kwargs):
    if diagnoser_type is None:
        from types import MethodType

        def dummy_method(self, *args, **kwargs):
            pass

        module.update_task_state = MethodType(dummy_method, module)
        module.step_done = MethodType(dummy_method, module)

        return module
    else:
        return _diagnoser.__dict__[diagnoser_type](module, *diagnoser_args, **diagnoser_kwargs)
