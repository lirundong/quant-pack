# -*- coding: utf-8 -*-

from . import diagnoser as _diagnoser
from ._registry import get_tasks

__all__ = ["get_diagnoser", "get_tasks"]


def get_diagnoser(diagnoser_type, module, *diagnoser_args, **diagnoser_kwargs):
    diagnoser = _diagnoser.__dict__[diagnoser_type](module, *diagnoser_args, **diagnoser_kwargs)

    return diagnoser
