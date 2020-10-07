from typing import Callable, List, Any
import numpy as np
import signal
import cv2
from .network_base import NetworkMeta
from .flow_base import Flow
from ..config import config
from ..environment import Environment

__all__ = ["register", "record", "register_handler", "track"]

def register(suite: str) -> Callable:
    def wrapper(func: Callable) -> Callable:
        def run(inst: "<Agent inst>") -> None:
            super(inst.__class__, inst).run(getattr(inst, suite))
        return run
    return wrapper

def record(func: Callable) -> Callable:
    def inner(inst: "<Agent inst>", frame: np.ndarray, state: Any = None) -> List:
        path = None
        if inst.config & (config.SAVE_FRAMES+config.SAVE_FLOW):
           if state is None:
              path = inst._frame_inventory.init_path
           else:
              path = inst._frame_inventory.path
           frame = inst.env.state.frame
           inst._frame_buffer.append(dict(path=path, frame=frame))
           if len(inst._frame_buffer) == inst.frame_buffer_size:
              for frame_ in inst._frame_buffer:
                  cv2.imwrite(frame_["path"], frame_["frame"])
              inst._frame_buffer.clear()
        return func(inst, frame, state), path
    return inner

def register_handler(unix_signals: List) -> Callable:
    def outer(handler: Callable) -> Callable:
        def inner(inst: "<Agent inst>", signal_id: int = None, frame: Any = None) -> None:
            for sig in unix_signals:
                signal.signal(sig, lambda x, y: handler(inst, x, y))
        return inner
    return outer

def track(network: NetworkMeta, config: bin = config.DEFAULT, flow: Flow = None) -> Callable:
    def outer(func: Callable) -> Callable:
        def inner(inst, env: Environment, network: NetworkMeta = network, config: bin = config,
                  flow: Flow = flow, **hyperparams) -> None:
            inst._params = dict(env=env, network=network, config=config, flow=flow)
            func(inst, env, network, config, flow, **hyperparams)
        return inner
    return outer
