import pytest
import csdl_alpha as csdl


@pytest.fixture
def recorder():
    """an active inline csdl recorder for the duration of a test"""
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield rec
    rec.stop()
