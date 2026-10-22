import inference


def test_package_imports():
    """The package imports and exposes a version string."""
    assert inference.__version__
