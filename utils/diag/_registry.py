# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict

# from .task import Task

__all__ = ["register", "get_tasks"]

_task_registry = dict()


def register(name):
    # TODO: why does this not work?

    def _do_register(task):
        # assert issubclass(task, Task)
        _task_registry[name] = task
        return task

    return _do_register


def get_tasks(diagnoser, task_conf):
    """ Build tasks from configuration and register to given diagnoser.

    Args:
        diagnoser (obj:`Diagnoser`): major observable diagnose registry;
        task_conf (list of dict): configuration in this format:

            - name: <name of task>
              type: <registered task type key>
              args:
                stage: required, {"pre_fwd", "fwd", "bkwd", "tensor"}
                other_args: ...
            - name: <other task name...>

    Returns:
        A list of built tasks, all registered to `diagnoser`.

    """
    from . import task as _task
    tasks = []

    for conf in task_conf:
        name = conf["name"]
        type = conf["type"]
        args = conf["args"]
        assert "stage" in args

        task_builder = _task.__dict__[type]
        task = task_builder(diagnoser, name, **args)
        tasks.append(task)

    diagnoser.register_hooks()

    return tasks
