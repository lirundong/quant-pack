# -*- coding: utf-8 -*-

from . import diagnoser as _diagnoser
from .registry import get_tasks

__all__ = ["get_diagnoser"]


def get_diagnoser(module, logger, diagnoser_type, tasks, diag_layers=None):
    tasks = get_tasks(tasks)
    diagnoser = _diagnoser.__dict__[diagnoser_type]

    return diagnoser(module, logger, tasks, diag_layers)
