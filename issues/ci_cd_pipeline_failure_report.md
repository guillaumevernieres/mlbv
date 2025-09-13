# CI/CD Pipeline Job Failure Report

## Issue Summary
The CI/CD Pipeline job has failed due to various errors that need to be addressed to ensure smooth functionality and compliance with standards.

## Errors:
1. **Python version in pyproject.toml:** Python 3.8 not supported (must be 3.9 or higher).
2. **Missing type annotations in icenet/distributed.py:** Lines 25, 38, 56, 85.
3. **Unsupported operand types** for `/` and `-` between Tensor and Module in `icenet/model.py` line 110.
4. **Return type mismatch** in `icenet/model.py` line 137.
5. **Argument type errors** in `icenet/data.py` line 123 for `select_data`.
6. **Missing library stubs** for yaml in `icenet/training.py`.
7. **Type assignment issues** and return value type mismatches in `icenet/training.py` (multiple lines).
8. **Missing type annotations** for variables and functions in multiple files.

## Suggested Fixes:
1. **Update** `pyproject.toml` to use `python_version = "3.9"` or higher.
2. **Add explicit type annotations** for all flagged functions and variables, e.g. `def my_func(x: int) -> None`.
3. **Ensure correct operand types** when combining Tensor and Module, e.g. use `module.weight` where needed.
4. **Update function return types** to match expected signatures, e.g. return `Tensor`, not `Optional[Tensor]` unless allowed.
5. **Update select_data** to accept `np.ndarray` or convert arguments as needed.
6. **Install library stubs:** `python3 -m pip install types-PyYAML` or `mypy --install-types`.
7. **Ensure type consistency** with assignments and scheduler returns, using correct type imports.
8. **Add missing variable and function type annotations,** e.g. `history: dict[str, list[float]] = {}` and `def foo() -> None`.