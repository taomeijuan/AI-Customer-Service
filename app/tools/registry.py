"""内置工具注册表。扫描 app.tools.plugins，不碰 agent / engine / main。"""

import importlib
import pkgutil

from app.tools.spec import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self.session_factory = None

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)


def scan_plugins(registry: ToolRegistry) -> None:
    """导入 app.tools.plugins 里每个模块（不含包本身）并调用 register。"""
    import app.tools.plugins as plugins_pkg

    prefix = plugins_pkg.__name__ + "."
    for info in pkgutil.iter_modules(plugins_pkg.__path__):
        if info.ispkg or info.name == "__init__":
            continue
        module = importlib.import_module(prefix + info.name)
        register = getattr(module, "register", None)
        if register is not None:
            register(registry)
