import time

import torch

profiler_events = []
is_enabled = False


def _start_fn(name, args=None):
    if is_enabled:
        profiler_events.append((name, args, time.time()))


def _stop_fn():
    if is_enabled:
        profiler_events.append((None, time.time()))


x = torch.rand((4, 4))

for _ in range(10):
    x.view_as(x)


def record_time(fn, name):
    for _ in range(100):
        fn()
    start = time.time()
    for _ in range(1000):
        fn()
    end = time.time()
    print(f"{name}:: {(end-start)} us")


def baseline():
    for _ in range(1000):
        x.view_as(x)


record_time(baseline, "Baseline")


def profiled_basic():
    for _ in range(1000):
        with torch.profiler.record_function("asdf"):
            x.view_as(x)


record_time(profiled_basic, "profiled_basic")


def fast_python():
    for _ in range(1000):
        _start_fn("asdf", {"source_attr": "fdfsfsdadfadf"})
        x.view_as(x)
        _stop_fn()


record_time(fast_python, "fast python")


def pybind_rf():
    for _ in range(1000):
        torch._C._profiler._record_function_fast_start_pybind("asdfx")
        x.view_as(x)
        torch._C._profiler._record_function_fast_stop_pybind()


record_time(pybind_rf, "pybind_rf")


def manual_bind_rf():
    for _ in range(1000):
        torch._C._profiler_manual._record_function_fast_start("asdfx")
        x.view_as(x)
        torch._C._profiler_manual._record_function_fast_stop()


record_time(manual_bind_rf, "manual_bind_rf")


def cpp_guarded_manual_bind_rf():
    for _ in range(1000):
        torch._C._profiler_manual._record_function_fast_start_checked("asdfx")
        x.view_as(x)
        torch._C._profiler_manual._record_function_fast_stop_checked()


record_time(cpp_guarded_manual_bind_rf, "cpp_guarded_manual_bind_rf")


def python_guarded_manual_bind_rf():
    for _ in range(1000):
        torch.profiler.profiler._record_function_fast_start("asdfx")
        x.view_as(x)
        torch.profiler.profiler._record_function_fast_stop()


record_time(python_guarded_manual_bind_rf, "python_guarded_manual_bind_rf")


def python_guarded_manual_bind_rf_unwrapped():
    for _ in range(1000):
        if torch.autograd.profiler._is_profiler_enabled:
            torch._C._profiler_manual._record_function_fast_start("asdfx")
        x.view_as(x)
        if torch.autograd.profiler._is_profiler_enabled:
            torch._C._profiler_manual._record_function_fast_stop()


record_time(
    python_guarded_manual_bind_rf_unwrapped, "python_guarded_manual_bind_rf_unwrapped"
)


class cm_record_function_fast_manual_bind:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.profiler.profiler._record_function_fast_start(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.profiler.profiler._record_function_fast_stop()


def python_guarded_cm_manual_bind_rf():
    for _ in range(1000):
        with cm_record_function_fast_manual_bind("asdfx"):
            x.view_as(x)


record_time(python_guarded_cm_manual_bind_rf, "python_guarded_cm_manual_bind_rf")


def manual_cm_rf():
    for _ in range(1000):
        with torch._C._profiler_manual._RecordFunctionFast("asdfx"):
            x.view_as(x)


record_time(manual_cm_rf, "manual_cm_rf")


def precompute_manual_cm_rf():
    cm = torch._C._profiler_manual._RecordFunctionFast("asdfx")
    for _ in range(1000):
        with cm:
            x.view_as(x)


record_time(precompute_manual_cm_rf, "precompute_manual_cm_rf")


def precompute_set_name_manual_cm_rf():
    cm = torch._C._profiler_manual._RecordFunctionFast("asdfx")
    for _ in range(1000):
        cm.set_name("asdfx")
        with cm:
            x.view_as(x)


record_time(precompute_set_name_manual_cm_rf, "precompute_set_name_manual_cm_rf")

with torch.profiler.profile() as prof:
    manual_bind_rf()

prof.export_chrome_trace("rfs_profile.json")
