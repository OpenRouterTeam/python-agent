"""Test package.

Present so `tests._fixtures` is importable as a module from test files. Without
it, `from tests._fixtures import ...` fails under pytest's default rootdir
handling and every test file falls back to hand-rolled stubs.
"""
