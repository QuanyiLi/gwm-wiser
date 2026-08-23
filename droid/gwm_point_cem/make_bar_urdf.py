"""make_bar_urdf: scoring URDF with the Robotiq subtree replaced by a rigid bar.

The bar's tip sits at d_tip = 0.1625 m along panda_link8 +z, the same point
as the closed 2F-85's fingertips; hover candidates are identical between the
gripper and bar runs, and only the 5 RAT renders differ. FrankaRobotRenderer
loads this URDF unchanged (no mimic chain; absent gripper joints are
skipped).

Run in the repo venv:  .venv/bin/python make_bar_urdf.py
"""

import xml.etree.ElementTree as ET

from config import BAR_URDF, URDF

BAR_LEN = 0.1625   # = PointerKinematics.d_tip measured on the closed 2F-85
BAR_RADIUS = 0.012

GRIPPER_LINKS = {
    "robotiq_arg2f_base_link",
    "left_outer_knuckle", "left_inner_knuckle", "left_outer_finger",
    "left_inner_finger", "left_inner_finger_pad",
    "right_outer_knuckle", "right_inner_knuckle", "right_outer_finger",
    "right_inner_finger", "right_inner_finger_pad",
}


def main() -> None:
    tree = ET.parse(str(URDF))
    root = tree.getroot()
    for link in list(root.findall("link")):
        if link.get("name") in GRIPPER_LINKS:
            root.remove(link)
    for joint in list(root.findall("joint")):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent in GRIPPER_LINKS or child in GRIPPER_LINKS:
            root.remove(joint)

    bar = ET.SubElement(root, "link", name="bar_link")
    vis = ET.SubElement(bar, "visual")
    ET.SubElement(vis, "origin", xyz=f"0 0 {BAR_LEN / 2}", rpy="0 0 0")
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=str(BAR_RADIUS), length=str(BAR_LEN))
    mat = ET.SubElement(vis, "material", name="bar_dark")
    ET.SubElement(mat, "color", rgba="0.25 0.25 0.28 1.0")
    joint = ET.SubElement(root, "joint", name="bar_joint", type="fixed")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(joint, "parent", link="panda_link8")
    ET.SubElement(joint, "child", link="bar_link")

    BAR_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(BAR_URDF))
    print(f"wrote {BAR_URDF}")


if __name__ == "__main__":
    main()
