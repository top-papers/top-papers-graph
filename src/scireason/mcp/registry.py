from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Iterable, get_type_hints

from .config import MCPServerConfig
from .specs import MCP_TOOL_ATTR, MCPToolMarker, MCPToolSpec


def _first_doc_line(obj: Any) -> str:
    # Use the first non-empty docstring line as the default tool description.
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def _humanize_name(name: str) -> str:
    # Fall back to a readable title when the decorator did not provide one.
    text = name.strip().replace("_", " ")
    return text[:1].upper() + text[1:] if text else "Tool"


def _validate_type_hints(func: Any) -> None:
    # Keep this strict so generated tool schemas stay predictable.
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    for param_name, param in sig.parameters.items():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            raise ValueError(f"{func.__module__}.{func.__name__}: variadic args are not supported for MCP tools")
        if param_name not in hints:
            raise ValueError(f"{func.__module__}.{func.__name__}: missing type hint for parameter '{param_name}'")
    if "return" not in hints:
        raise ValueError(f"{func.__module__}.{func.__name__}: missing return type hint")


def _build_spec(func: Any, marker: MCPToolMarker) -> MCPToolSpec:
    # Build the final runtime spec from decorator metadata plus function metadata.
    _validate_type_hints(func)

    description = (marker.description or _first_doc_line(func)).strip()
    if not description:
        raise ValueError(f"{func.__module__}.{func.__name__}: MCP tool must define a description or docstring")

    name = (marker.name or func.__name__).strip()
    if not name:
        raise ValueError(f"{func.__module__}.{func.__name__}: MCP tool name resolved to empty string")

    title = (marker.title or _humanize_name(name)).strip()
    meta = dict(marker.meta)
    meta.setdefault("toolset", marker.toolset)
    meta.setdefault("read_only", marker.read_only)
    meta.setdefault("registration_mode", "auto")
    meta.setdefault("origin_module", func.__module__)

    return MCPToolSpec(
        func=func,
        toolset=marker.toolset,
        name=name,
        title=title,
        description=description,
        meta=meta,
        structured_output=marker.structured_output,
        read_only=marker.read_only,
        enabled_by_default=marker.enabled_by_default,
        origin_module=func.__module__,
    )


def _iter_toolset_module_names(package_name: str) -> Iterable[str]:
    # Import every plain module under the toolsets package.
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if not package_path:
        return []
    return (
        module_info.name
        for module_info in pkgutil.walk_packages(package_path, prefix=f"{package_name}.")
        if not module_info.ispkg
    )


@dataclass
class MCPToolRegistry:
    specs_by_name: dict[str, MCPToolSpec] = field(default_factory=dict)

    @classmethod
    def discover(
        cls,
        *,
        package_name: str = "scireason.mcp.toolsets",
        extra_modules: Iterable[str] = (),
    ) -> "MCPToolRegistry":
        # Discover decorated functions from the default toolsets package
        # and from any extra modules listed in the env config.
        registry = cls()
        module_names = list(_iter_toolset_module_names(package_name))
        module_names.extend(m for m in extra_modules if m)

        for module_name in sorted(dict.fromkeys(module_names)):
            module = importlib.import_module(module_name)
            for _, member in inspect.getmembers(module, inspect.isfunction):
                if member.__module__ != module.__name__:
                    continue
                marker = getattr(member, MCP_TOOL_ATTR, None)
                if not isinstance(marker, MCPToolMarker):
                    continue
                spec = _build_spec(member, marker)
                registry.add(spec)
        return registry

    def add(self, spec: MCPToolSpec) -> None:
        # Duplicate tool names are hard to debug later, so fail early here.
        if spec.name in self.specs_by_name:
            existing = self.specs_by_name[spec.name]
            raise ValueError(
                f"Duplicate MCP tool name '{spec.name}' in {existing.origin_module} and {spec.origin_module}"
            )
        self.specs_by_name[spec.name] = spec

    def all_specs(self) -> list[MCPToolSpec]:
        return [self.specs_by_name[name] for name in sorted(self.specs_by_name)]

    def validate_selection(self, config: MCPServerConfig) -> None:
        # Fail fast if env-based selection refers to unknown tools or toolsets.
        unknown_tools = sorted(config.tools - frozenset(self.specs_by_name))
        unknown_toolsets = sorted(config.toolsets - frozenset(self.toolsets()))
        if unknown_tools or unknown_toolsets:
            parts: list[str] = []
            if unknown_tools:
                parts.append(f"unknown tools: {', '.join(unknown_tools)}")
            if unknown_toolsets:
                parts.append(f"unknown toolsets: {', '.join(unknown_toolsets)}")
            raise ValueError("Invalid MCP selection config: " + "; ".join(parts))

    def selected_specs(self, config: MCPServerConfig) -> list[MCPToolSpec]:
        # Final filtering step before FastMCP registration.
        selected: list[MCPToolSpec] = []
        for spec in self.all_specs():
            if config.read_only and not spec.read_only:
                continue
            if spec.name in config.disabled_tools:
                continue
            if config.has_selection_filters:
                if spec.name not in config.tools and spec.toolset not in config.toolsets:
                    continue
            elif not spec.enabled_by_default:
                continue
            selected.append(spec)
        return selected

    def toolsets(self) -> list[str]:
        return sorted({spec.toolset for spec in self.specs_by_name.values()})

    def summary(self, config: MCPServerConfig) -> dict[str, Any]:
        # Small diagnostic payload used by doctor_tool.
        selected = self.selected_specs(config)
        return {
            "discovered_tools": len(self.specs_by_name),
            "selected_tools": len(selected),
            "toolsets": self.toolsets(),
            "selected_tool_names": [spec.name for spec in selected],
            "read_only_mode": config.read_only,
            "selection_filters_active": config.has_selection_filters,
        }
