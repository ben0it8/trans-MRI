from functools import partial
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler


def scheduler_lin_cyc(start, stop, cycle_size):
    return partial(LinearCyclicalScheduler, param_name='lr', 
                  start_value=1e-4, end_value=5e-1, cycle_size=cycle_size,
                    save_history=True)

