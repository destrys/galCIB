import galCIB


def test_can_import_galCIB():
    assert "CIBModel" in dir(galCIB)
