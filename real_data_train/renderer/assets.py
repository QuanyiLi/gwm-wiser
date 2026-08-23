"""Robot URDF assets for the shared Franka renderer.

No public combined arm+gripper URDF exists, so this
module fetches the three public sources and welds per-rig URDFs:

- Panda arm: haosulab/ManiSkill ``mani_skill/assets/robots/panda/panda_v2.urdf``
  (sparse clone), Franka hand stripped — the real DROID rig arm.
- FR3 arm: BolunDai0216/FR3Env ``FR3Env/robots/fr3.urdf``, hand stripped —
  the MolmoBot / MolmoSpaces / TiPToP hardware rig arm.
- Robotiq 2F-85: haosulab/ManiSkill-Robotiq_2F ``robotiq_2f_85.urdf``
  (SAPIEN-tuned, no mimic tags: six independent revolute joints).

The welded URDF attaches the gripper base to the arm flange link with a fixed
joint (standard pattern, cf. ManiSkill-XArm6). Mesh references are rewritten
to absolute paths, so the generated file is machine-local (it lives under the
gitignored data/assets tree). ``provenance()`` captures the source commits for
the rendered-tree metadata.
"""

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

REPOS = {
    "ManiSkill": ("https://github.com/haosulab/ManiSkill.git",
                  ["mani_skill/assets/robots/panda"]),
    "ManiSkill-Robotiq_2F": ("https://github.com/haosulab/ManiSkill-Robotiq_2F.git",
                             None),
    "FR3Env": ("https://github.com/BolunDai0216/FR3Env.git", None),
}

ARM_URDF = {
    "panda": "ManiSkill/mani_skill/assets/robots/panda/panda_v2.urdf",
    "fr3": "FR3Env/FR3Env/robots/fr3.urdf",
}
GRIPPER_URDF = "ManiSkill-Robotiq_2F/robotiq_2f_85.urdf"

# Flange link per arm (ISO 9409-1-50-4-M6 on both rigs, so one mount serves
# both). The mount transform is a smoke-stage default; the pre-flight overlay
# gate is what validates it per source before large-scale rendering.
FLANGE_LINK = {"panda": "panda_link8", "fr3": "fr3_link8"}
MOUNT_XYZ = "0 0 0.004"   # Robotiq coupling ring stand-off
MOUNT_RPY = "0 0 0"


def _run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def ensure_source_repos(assets_root: Path) -> None:
    src = Path(assets_root) / "src"
    src.mkdir(parents=True, exist_ok=True)
    for name, (url, sparse) in REPOS.items():
        dest = src / name
        if (dest / ".git").is_dir():
            continue
        if sparse:
            _run(["git", "clone", "--depth", "1", "--filter=blob:none",
                  "--sparse", url, str(dest)])
            _run(["git", "sparse-checkout", "set", *sparse], cwd=dest)
        else:
            _run(["git", "clone", "--depth", "1", url, str(dest)])


def provenance(assets_root: Path) -> dict:
    src = Path(assets_root) / "src"
    out = {}
    for name in REPOS:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=src / name,
            capture_output=True, text=True,
        )
        out[name] = head.stdout.strip()
    return out


def _absolutize_meshes(root: ET.Element, urdf_dir: Path,
                       package_root: Path = None) -> None:
    """Rewrite relative and package:// mesh refs to absolute paths.

    FR3Env uses ``package://robots/...`` with the package root at the
    directory holding ``robots/`` (the inner FR3Env package dir).
    """
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn or fn.startswith("/"):
            continue
        if fn.startswith("package://"):
            if package_root is None:
                raise ValueError(f"package:// mesh but no package root: {fn}")
            mesh.set("filename",
                     str((package_root / fn[len("package://"):]).resolve()))
        else:
            mesh.set("filename", str((urdf_dir / fn).resolve()))


def _links_reachable_from(root: ET.Element, start: str) -> set:
    """Links in the subtree hanging below `start` (via joint parent->child)."""
    children = {}
    for joint in root.iter("joint"):
        p = joint.find("parent").get("link")
        c = joint.find("child").get("link")
        children.setdefault(p, []).append(c)
    seen, stack = set(), [start]
    while stack:
        link = stack.pop()
        for c in children.get(link, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def build_welded_urdf(arm: str, assets_root: Path) -> Path:
    """Weld <arm> + Robotiq 2F-85 into data/assets/<arm>_robotiq.urdf."""
    assets_root = Path(assets_root)
    ensure_source_repos(assets_root)
    out_path = assets_root / f"{arm}_robotiq.urdf"

    arm_path = assets_root / "src" / ARM_URDF[arm]
    grip_path = assets_root / "src" / GRIPPER_URDF
    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()
    grip_root = ET.parse(grip_path).getroot()
    _absolutize_meshes(
        arm_root, arm_path.parent,
        package_root=(assets_root / "src" / "FR3Env" / "FR3Env"
                      if arm == "fr3" else None),
    )
    _absolutize_meshes(grip_root, grip_path.parent)

    flange = FLANGE_LINK[arm]
    # Strip everything below the flange (Franka hand / fingers / tcp frames).
    doomed = _links_reachable_from(arm_root, flange)
    for link in list(arm_root.findall("link")):
        if link.get("name") in doomed:
            arm_root.remove(link)
    for joint in list(arm_root.findall("joint")):
        if joint.find("child").get("link") in doomed:
            arm_root.remove(joint)

    # Import all gripper links/joints/materials, then weld.
    grip_base = None
    grip_links = {l.get("name") for l in grip_root.findall("link")}
    grip_children = {j.find("child").get("link")
                     for j in grip_root.findall("joint")}
    roots = grip_links - grip_children
    assert len(roots) == 1, f"gripper URDF has {len(roots)} root links: {roots}"
    grip_base = roots.pop()
    for tag in ("material", "link", "joint"):
        for el in grip_root.findall(tag):
            arm_root.append(el)

    weld = ET.SubElement(arm_root, "joint",
                         {"name": "gripper_mount", "type": "fixed"})
    ET.SubElement(weld, "parent", {"link": flange})
    ET.SubElement(weld, "child", {"link": grip_base})
    ET.SubElement(weld, "origin", {"xyz": MOUNT_XYZ, "rpy": MOUNT_RPY})

    arm_root.set("name", f"{arm}_robotiq")
    arm_tree.write(out_path)
    return out_path
