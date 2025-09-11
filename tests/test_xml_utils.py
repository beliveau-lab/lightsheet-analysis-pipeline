from scripts.utils.xml_utils import get_channels, check_xml_stage


def test_get_channels_handles_missing_file():
    assert get_channels("/path/does/not/exist.xml") == [0]


def test_check_xml_stage_handles_missing_file():
    assert check_xml_stage("/path/does/not/exist.xml", "pairwise") is False

