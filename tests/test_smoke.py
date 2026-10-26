import app


def test_app_package_is_importable() -> None:
    assert app.__name__ == "app"
