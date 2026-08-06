from pathlib import Path
from xml.etree import ElementTree


def test_v2_cube_orientation_scene_defines_closeup_training_camera():
    scene_path = (
        Path(__file__).parents[1]
        / "src/orca_sim/scenes/v2/scene_right_cube_orientation.xml"
    )
    root = ElementTree.parse(scene_path).getroot()

    camera = root.find(".//camera[@name='closeup']")

    assert camera is not None
    assert camera.attrib["target"] == "task_cube"
