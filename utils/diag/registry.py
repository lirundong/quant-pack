# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict

__all__ = ["register", "get_tasks"]

_task_builders = dict()


def register(name):

    def _do_register(func):
        _task_builders[name] = func
        return func

    return _do_register


def get_tasks(task_name_registry):
    tasks = defaultdict(OrderedDict)

    for stage_name, tasks_names in task_name_registry.items():
        assert stage_name in ("pre_fwd", "fwd", "bkwd")
        for task_name in tasks_names:
            assert task_name in _task_builders, f"not registered task: {task_name}"
            task_builder = _task_builders[task_name]
            tasks[stage_name].update([(task_name, task_builder), ])

    return tasks
