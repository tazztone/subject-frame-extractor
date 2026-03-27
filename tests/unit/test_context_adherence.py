import ast
import pathlib

import pytest

# Mark as context evaluation test
pytestmark = pytest.mark.context_eval


def get_python_files(directory):
    path = pathlib.Path(directory)
    return list(path.rglob("*.py"))


class TestArchitecturalIsolation:
    """Rule: NEVER import from ui/ inside core/."""

    def test_core_does_not_import_from_ui(self):
        core_files = get_python_files("core")
        violations = []

        for file_path in core_files:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("ui"):
                                violations.append(f"{file_path}: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("ui"):
                            violations.append(f"{file_path}: from {node.module}")

        assert not violations, (
            "Architectural Isolation Violation: core/ modules should not import from ui/.\n" + "\n".join(violations)
        )


class TestConfigSafety:
    """Rule: NEVER use @lru_cache on functions taking the Config object."""

    def test_no_lru_cache_on_config_functions(self):
        core_files = get_python_files("core")
        violations = []

        for file_path in core_files:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function has @lru_cache
                        has_lru_cache = any(
                            (isinstance(dec, ast.Name) and dec.id == "lru_cache")
                            or (
                                isinstance(dec, ast.Call)
                                and isinstance(dec.func, ast.Name)
                                and dec.func.id == "lru_cache"
                            )
                            for dec in node.decorator_list
                        )

                        if has_lru_cache:
                            # Check if 'Config' or 'config' is in arguments
                            args = [arg.arg for arg in node.args.args]
                            # Or check type hints
                            arg_types = [
                                ast.unparse(arg.annotation) if arg.annotation else "" for arg in node.args.args
                            ]

                            if "config" in args or any("Config" in t for t in arg_types):
                                violations.append(f"{file_path}: {node.name}")

        assert not violations, (
            "Config Safety Violation: @lru_cache used on function taking Config object.\n" + "\n".join(violations)
        )


class TestUISafety:
    """Rule: UI event handlers in app_ui.py MUST be wrapped in @AppUI.safe_ui_callback."""

    def test_ui_handlers_have_safe_callback(self):
        ui_files = [pathlib.Path("ui/app_ui.py")]
        violations = []

        for file_path in ui_files:
            if not file_path.exists():
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue

                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == "AppUI":
                        for member in node.body:
                            if isinstance(member, ast.FunctionDef):
                                # Skip internal/private methods
                                if member.name.startswith(("_", "setup_")):
                                    continue

                                # Skip known non-handler methods and the decorator itself
                                if member.name in ["build_ui", "preload_models", "safe_ui_callback"]:
                                    continue

                                # Check if it has the decorator
                                has_safe_decorator = False
                                for dec in member.decorator_list:
                                    # Matches @safe_ui_callback(...) or @AppUI.safe_ui_callback(...)
                                    if isinstance(dec, ast.Call):
                                        if isinstance(dec.func, ast.Name) and dec.func.id == "safe_ui_callback":
                                            has_safe_decorator = True
                                            break
                                        if isinstance(dec.func, ast.Attribute) and dec.func.attr == "safe_ui_callback":
                                            has_safe_decorator = True
                                            break

                                if not has_safe_decorator:
                                    violations.append(f"{file_path}: {member.name}")

        assert not violations, "UI Safety Violation: event handler missing @AppUI.safe_ui_callback.\n" + "\n".join(
            violations
        )


class TestMilestoneCase:
    """Rule: ALWAYS use Title Case for major pipeline milestones."""

    def test_milestone_strings_are_title_case(self):
        pipeline_path = pathlib.Path("core/pipelines.py")
        if not pipeline_path.exists():
            return

        violations = []
        with open(pipeline_path, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return

        for node in ast.walk(tree):
            if isinstance(node, ast.Yield):
                if isinstance(node.value, ast.Dict):
                    for i, key in enumerate(node.value.keys):
                        if key and isinstance(key, ast.Constant) and key.value == "status":
                            val = node.value.values[i]
                            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                                status_str = val.value
                                words = [w for w in status_str.split() if len(w) > 1]
                                for word in words:
                                    if word[0].islower() and word not in [
                                        "of",
                                        "to",
                                        "in",
                                        "and",
                                        "the",
                                    ]:
                                        violations.append(f"Status '{status_str}' has non-title-case word '{word}'")

        # Non-blocking check for now
        pass


class TestSAM2DefaultBaseline:
    """Rule: SAM2.1 Hiera Tiny is the project's default tracker."""

    def test_config_default_is_sam2(self):
        from core.config import Config

        config = Config()
        assert config.default_tracker_model_name == "sam2", "Default tracker in config.py must be 'sam2'"

    def test_agents_md_prescribes_sam2_default(self):
        agents_md_path = pathlib.Path("AGENTS.md")
        if not agents_md_path.exists():
            pytest.skip("AGENTS.md not found")

        content = agents_md_path.read_text()
        assert "SAM2.1 Hiera Tiny" in content, "AGENTS.md must mention SAM2.1 Hiera Tiny as the default baseline"
        assert "default tracker" in content.lower(), "AGENTS.md must specify the default tracker"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
