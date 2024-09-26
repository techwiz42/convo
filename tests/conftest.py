def pytest_addoption(parser):
    parser.addoption("--qa-cli-version", action="store", default="v1", help="Version of qa_cli to test")

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "qa_cli_version(version): mark test to run only for specific qa_cli version"
    )
