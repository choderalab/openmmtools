import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu_ci: mark test as useful to run on GPU")
    config.addinivalue_line("markers", "gpu_needed: mark test as GPU required")