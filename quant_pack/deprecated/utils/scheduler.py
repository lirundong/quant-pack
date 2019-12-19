# -*- coding: utf-8 -*-

import collections
import numbers

__all__ = ["IterationScheduler"]

ScheduledOptCfg = collections.namedtuple(
    "ScheduledOptCfg",
    field_names=["name", "start_iter", "end_iter", "iter_or_epoch"],
)

ScheduledVariableCfg = collections.namedtuple(
    "ScheduledVariableCfg",
    field_names=["name", "init_value", "target_value",
                 "warmup_start_iter", "warmup_done_iter", "terminate_iter",
                 "iter_or_epoch"],
)


class IterationScheduler:
    # TODO:
    #  1. type-hinting
    #  2. currently all start/end/terminate_iters are actually passed by epochs,
    #     modify the semantic to make config more clear

    def __init__(self,
                 named_opt_schedulers,
                 scheduled_opt_cfgs,
                 scheduled_variable_cfgs,
                 iters_per_epoch,
                 quant_start_iter,
                 total_iters,
                 dynamic_variable_scale=1.0):

        self.opt_schedulers = collections.OrderedDict(named_opt_schedulers)
        self.iters_per_epoch = iters_per_epoch
        self.quant_start_iter = quant_start_iter
        self.total_iters = total_iters
        self.dynamic_variable_scale = dynamic_variable_scale

        self._add_scheduled_opt_cfgs(*scheduled_opt_cfgs)
        self._add_scheduled_variable_cfgs(*scheduled_variable_cfgs)

    def _add_scheduled_opt_cfgs(self, *opt_cfgs: ScheduledOptCfg):
        """Store optimizer schedules and their activated periods.

        Fields of scheduled optimizers `opt_cfg`s are:

          ["name", "start_iter", "end_iter", "iter_or_epoch"]

        note that we convert all start/end_epochs to iters if opt_schedule get
        `iters_or_epochs=="epoch"`, but only invoke its `step()` by number of
        epochs.
        """
        self.per_iter_opt_schedules = []
        self.per_epoch_opt_schedules = []
        for opt_cfg in opt_cfgs:
            assert isinstance(opt_cfg, (list, tuple)) and len(opt_cfg) == 4
            assert opt_cfg[0] in self.opt_schedulers
            assert opt_cfg[3] == "iter" or opt_cfg[3] == "epoch"
            if isinstance(opt_cfg, tuple):
                opt_cfg = list(opt_cfg)
            # TODO: make this config semantic more clear
            opt_cfg[1] *= self.iters_per_epoch
            if opt_cfg[2] != -1:
                opt_cfg[2] *= self.iters_per_epoch
            else:
                # this schedule terminates at training ending
                opt_cfg[2] = self.total_iters + 1
            try:
                opt_schedule = ScheduledOptCfg(*opt_cfg)
            except TypeError as e:
                raise RuntimeError(f"When building {ScheduledOptCfg.__name__} from `{opt_cfg}`: {e}")
            if opt_schedule.iter_or_epoch == "iter":
                self.per_iter_opt_schedules.append(opt_schedule)
            else:
                self.per_epoch_opt_schedules.append(opt_schedule)
        if len(self.per_iter_opt_schedules) > 0:
            last_iters = [self.opt_schedulers[opt.name].last_epoch
                          for opt in self.per_iter_opt_schedules]
        elif len(self.per_epoch_opt_schedules) > 0:
            last_iters = [self.opt_schedulers[opt.name].last_epoch * self.iters_per_epoch
                          for opt in self.per_epoch_opt_schedules]
        else:
            last_iters = [0, ]
        assert len(set(last_iters)) == 1, f"`scheduled_opts` get different `last_epoch` field"
        self.last_iter = last_iters[0]

    def _add_scheduled_variable_cfgs(self, *var_cfgs: ScheduledVariableCfg):
        """Add var_cfg schedules by descriptions in configuration files.

        The valid config formats of scheduled_variables are:
        *  a yaml list of 6 fields: [name, init_value, target_value,
           warmup_start_step, warmup_done_step, terminate_step].

           This kind of variables are linearly warmed-up from `init_value` to
           `target_value` during the interval of [warmup_start_step,
           warmup_done_step], then reduce to 0 immediately at `terminate_step`.
           Note that one var_cfg could have multiple such descriptions, each
           description forms one schedule period;
        *  a yaml list of 2 fields: [name, ref_var_name]. The var_cfg will be
           scaled by the ratio of $\frac{*ref_var_name}{\sum{*ref_var_names}}$;
        *  a yaml list of 2 fields: [name, val]
        """
        self.linear_warmup_variables = collections.defaultdict(list)
        self.const_variables = collections.OrderedDict()
        self.dynamic_scaling_variables = collections.OrderedDict()
        self.scaling_reference_names = set()
        self.variable_states = collections.OrderedDict()

        for var_cfg in var_cfgs:
            assert isinstance(var_cfg, (tuple, list))
            if len(var_cfg) == 2:
                if isinstance(var_cfg[1], str):
                    self.dynamic_scaling_variables[var_cfg[0]] = var_cfg[1]
                    self.scaling_reference_names.add(var_cfg[1])
                elif isinstance(var_cfg[1], numbers.Number):
                    self.const_variables[var_cfg[0]] = var_cfg[1]
                else:
                    raise ValueError(f"unknown var_cfg schedule: {var_cfg}")
            elif len(var_cfg) == 7:
                if isinstance(var_cfg, tuple):
                    var_cfg = list(var_cfg)
                if var_cfg[6] == "epoch":
                    # convert "warmup_start_iter", "warmup_done_iter" and
                    # "terminate_iter" (if not -1) from epochs to iters
                    var_cfg[3] *= self.iters_per_epoch
                    var_cfg[4] *= self.iters_per_epoch
                    if var_cfg[5] != -1:
                        var_cfg[5] *= self.iters_per_epoch
                if var_cfg[5] == -1:
                    # this schedule terminates at training ending
                    var_cfg[5] = self.total_iters + 1
                try:
                    scheduled_variable = ScheduledVariableCfg(*var_cfg)
                except TypeError as e:
                    raise RuntimeError(f"When building {ScheduledVariableCfg.__name__} from `{var_cfg}`: {e}")
                assert scheduled_variable.warmup_start_iter is not None, \
                    "you are using deprecated configuration for const variables, " \
                    "please convert to 2 fields format of: [name, val]"
                self.linear_warmup_variables[scheduled_variable.name].append(scheduled_variable)
            else:
                raise ValueError(f"Invalid schedule_variable config: {var_cfg}")

        for var_name, var_schedules in self.linear_warmup_variables.items():
            # sort and check schedules are not overlapping
            sorted_schedules = sorted(var_schedules, key=lambda x: x.warmup_start_iter)
            for i in range(len(sorted_schedules) - 1):
                assert sorted_schedules[i].terminate_iter < sorted_schedules[i + 1].warmup_start_iter
            self.linear_warmup_variables[var_name] = sorted_schedules

    @property
    def quant_enabled(self):
        return self.quant_start_iter <= self.last_iter

    @property
    def do_calibration(self):
        return self.quant_start_iter == self.last_iter

    def state_dict(self):
        scheduler_states = collections.OrderedDict()
        for k, v in self.opt_schedulers.items():
            scheduler_state = v.state_dict()
            scheduler_states[k] = scheduler_state
        self_states = collections.OrderedDict()
        self_states.update({k: v for k, v in self.__dict__.items() if
                            (not k.startswith("__") and k != "opt_schedulers")})
        self_states["opt_schedulers"] = scheduler_states
        return self_states

    def load_state_dict(self, state_dict):
        scheduler_states = state_dict["opt_schedulers"]
        for k, v in self.opt_schedulers.items():
            v.load_state_dict(scheduler_states[k])
        state_dict.pop("opt_schedulers")
        self.__dict__.update(state_dict)

    def _get_variable_state(self, var_name, **ref_vals):
        if var_name in self.variable_states:
            # linearly-warmup variables
            return self.variable_states[var_name]

        elif var_name in self.const_variables:
            # const variables
            return self.const_variables[var_name]

        elif var_name in self.dynamic_scaling_variables:
            # dynamically scaling by other reference variables
            var_ref_val = ref_vals[self.dynamic_scaling_variables[var_name]]
            total_ref_val = sum(ref_vals[ref_name] for ref_name in self.scaling_reference_names)
            return var_ref_val / total_ref_val * self.dynamic_variable_scale

        else:
            raise ValueError(f"unregistered variable name `{var_name}`")

    def get_scheduled_variables(self, *var_names, **ref_vals):
        ret = []
        for var_name in var_names:
            ret.append(self._get_variable_state(var_name, **ref_vals))
        return ret

    def _update_variable_states(self):

        def _in_schedule_interval(_schedule):
            return _schedule.warmup_start_iter <= self.last_iter < _schedule.iterminate_iter

        def _activated_schedule(_schedules):
            _ret_schedule = None
            for _schedule in _schedules:
                if _in_schedule_interval(_schedule):
                    _ret_schedule = _schedule
                    break
            return _ret_schedule

        def _in_schedule_warmup(_schedule):
            return _schedule.warmup_start_iter <= self.last_iter < _schedule.warmup_done_iter

        for var_name, var_schedules in self.linear_warmup_variables.items():
            schedule = _activated_schedule(var_schedules)
            if schedule is not None:
                if _in_schedule_warmup(schedule):
                    delta = (schedule.target_value - schedule.init_value) \
                            / (schedule.warmup_done_iter - schedule.warmup_start_iter + 1)
                    warmed_iters = self.last_iter - schedule.warmup_start_iter
                    self.variable_states[var_name] = schedule.init_value + delta * warmed_iters
                else:
                    self.variable_states[var_name] = schedule.target_value
            else:
                self.variable_states[var_name] = 0.

    def _update_opt_schedulers(self):

        def _in_schedule_interval(_schedule):
            return _schedule.start_iter <= self.last_iter < _schedule.end_iter

        def _check_and_take_step(_schedules):
            for _schedule in _schedules:
                if _in_schedule_interval(_schedule):
                    _scheduler = self.opt_schedulers[_schedule.name]
                    _scheduler.step()

        if self.last_iter > 0 and self.last_iter % self.iters_per_epoch == 0:
            _check_and_take_step(self.per_epoch_opt_schedules)

        _check_and_take_step(self.per_iter_opt_schedules)

    def step(self, iteration=None):
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self._update_variable_states()
        self._update_opt_schedulers()
