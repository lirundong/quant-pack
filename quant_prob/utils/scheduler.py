# -*- coding: utf-8 -*-

import collections
from logging import getLogger

__all__ = ["IterationScheduler"]

ScheduledVariable = collections.namedtuple(
    "ScheduledVariable",
    field_names=["name", "init_value", "target_value",
                 "warmup_start_step", "warmup_done_step", "terminate_step"],
)


class IterationScheduler(object):
    def __init__(self, optimizer, milestones, dataset_size, batch_size, total_iters,
                 warmup_epochs=0, warmup_lr=0, world_size=1, gamma=0.1,
                 last_iter=-1, enable_quant_at=None, scheduled_variables=None,
                 verbose=False):
        batch_size *= world_size
        iters_per_epoch = dataset_size // batch_size
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        if warmup_lr > 0:
            self.base_lrs = [warmup_lr, ] * len(optimizer.param_groups)
        else:
            self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.target_lrs = list(map(lambda group: group['initial_lr'] * world_size, optimizer.param_groups))

        if not isinstance(milestones, collections.Iterable):
            milestones = [milestones, ]

        self.milestones = [m * iters_per_epoch for m in milestones]
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = total_iters
        self.world_size = world_size
        self.gamma = gamma
        self.optimizer = optimizer
        self.last_iter = last_iter

        if enable_quant_at == "begin":
            self.enable_quant_intervals = [(1, self.total_iters + 1), ]  # step starts from 1
        elif enable_quant_at == "segmented":
            self.enable_quant_intervals = []
        elif enable_quant_at is not None:
            self.enable_quant_intervals = [(enable_quant_at * self.iters_per_epoch, self.total_iters + 1), ]
        else:
            self.enable_quant_intervals = []

        self.variable_schedules = collections.defaultdict(list)
        self.variable_states = collections.OrderedDict()
        self.const_variables = collections.OrderedDict()
        if scheduled_variables is not None:
            self.add_scheduled_variables(*scheduled_variables)

        self._update_next_milestone()
        self.step(last_iter + 1)

        if verbose:
            logger = getLogger("global")
            logger.info(str(self))

    def _update_next_milestone(self):
        remain_milestones = [m for m in self.milestones if m > self.last_iter]
        self.next_milestone = min(remain_milestones) if len(remain_milestones) > 0 else None

    def __str__(self):
        info_tokens = [f"milestone iters: {self.milestones}", ]
        if self.warmup_iters > 0:
            info_tokens += [f"warmup iters: {self.warmup_iters}",
                            f"LR before warmup: {self.base_lrs}",
                            f"LR after warmup: {self.target_lrs}"]
        if len(self.enable_quant_intervals) > 0:
            info_tokens += [f"enable quantization at iter {self.enable_quant_intervals}"]

        return "\n".join(info_tokens)

    def add_scheduled_variables(self, *variables):
        for variable in variables:
            assert isinstance(variable, (tuple, list))
            # fields: name, init_value, target_value, warmup_start_step, warmup_done_step, terminate_step
            try:
                scheduled_variable = ScheduledVariable(*variable)
            except TypeError as e:
                raise RuntimeError(f"When building {ScheduledVariable.__name__} from `{variable}`: {e}")
            if scheduled_variable.warmup_start_step is not None:
                terminate_step = scheduled_variable.terminate_step * self.iters_per_epoch \
                    if scheduled_variable.terminate_step != -1 \
                    else self.total_iters + 1
                scheduled_variable = scheduled_variable._replace(
                    warmup_start_step=scheduled_variable.warmup_start_step * self.iters_per_epoch,
                    warmup_done_step=scheduled_variable.warmup_done_step * self.iters_per_epoch,
                    terminate_step=terminate_step,
                )
                self.variable_schedules[scheduled_variable.name].append(scheduled_variable)
                self.enable_quant_intervals.append((scheduled_variable.warmup_start_step, scheduled_variable.terminate_step, ))
            else:
                self.const_variables[scheduled_variable.name] = \
                    (scheduled_variable.init_value, scheduled_variable.target_value, )

        for k, v in self.variable_schedules.items():
            self.variable_schedules[k] = sorted(v, key=lambda x: x.warmup_start_step)
        if len(self.enable_quant_intervals) > 0:
            self.enable_quant_intervals.sort(key=lambda x: x[0])

    def update_scheduled_variables(self):
        for var_name, schedules in self.variable_schedules.items():
            if self.quant_enabled:
                for schedule in schedules:
                    if self.last_iter < schedule.warmup_start_step or schedule.terminate_step <= self.last_iter:
                        continue
                    elif schedule.warmup_start_step <= self.last_iter < schedule.warmup_done_step:  # warmup
                        delta = (schedule.target_value - schedule.init_value) \
                                / (schedule.warmup_done_step - schedule.warmup_start_step + 1)
                        warmed_steps = self.last_iter - schedule.warmup_start_step
                        self.variable_states[var_name] = schedule.init_value + delta * warmed_steps
                    else:  # target value
                        self.variable_states[var_name] = schedule.target_value
            else:
                self.variable_states[var_name] = 0.
        for var_name, consts in self.const_variables.items():
            if self.quant_enabled:
                self.variable_states[var_name] = consts[1]  # target value
            else:
                self.variable_states[var_name] = consts[0]  # init value

    def get_scheduled_variables(self, *var_names):
        ret = []
        for var_name in var_names:
            ret.append(self.get_scheduled_variable(var_name))
        return ret

    def get_scheduled_variable(self, var_name):
        return self.variable_states[var_name]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        if state_dict["iters_per_epoch"] != self.iters_per_epoch:
            logger = getLogger("global")
            logger.warning(f"`state_dict` get different `iters_per_epoch` than this experiment, "
                           f"so we only recover the actual number of gone iterations")
            last_iter = state_dict["last_iter"] // state_dict["iters_per_epoch"] * self.iters_per_epoch
            self.step(last_iter)
        else:
            self.__dict__.update(state_dict)

    def step(self, iteration=None):
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration

        self.update_scheduled_variables()

        # linear warmup LR
        if self.last_iter < self.warmup_iters:
            for param, base_lr, target_lr in zip(self.optimizer.param_groups, self.base_lrs, self.target_lrs):
                lr_delta = (target_lr - base_lr) / self.warmup_iters
                lr = base_lr + lr_delta * self.last_iter
                param["lr"] = lr
            return

        # scale LR by world_size
        if self.last_iter == self.warmup_iters and self.world_size > 1:
            for param_group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                param_group["lr"] = target_lr
            return

        # LR decay
        if self.next_milestone is None or self.last_iter < self.next_milestone:
            return

        self._update_next_milestone()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma

    @property
    def quant_enabled(self):
        if len(self.enable_quant_intervals) > 1 and \
                any(x[0] <= self.last_iter < x[1] for x in self.enable_quant_intervals):
            return True
        else:
            return False

    @property
    def do_calibration(self):
        if len(self.enable_quant_intervals) > 1 and self.enable_quant_intervals[0][0] == self.last_iter:
            return True
        else:
            return False
