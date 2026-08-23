"""The 14 scene-6 tasks and how they map onto the candidate pools.

Instructions / judge rules are copied from droid_sim_ret/tasks/{refer6,place}_tasks.sh
(the instruction is never shown to V-JEPA 2-AC; it only names which candidates
count as correct). Cluster ids follow droid_sim_ret/PARSING.md section 3 and
proposals_index.json; the judged rigid-object names are the Isaac prims.
"""

PICK_TASKS = [
    # tag, instruction, target cluster in scene6_rev2, rigid object judged
    ("fruit", "pick up the fruit", "object_4", "_11_banana"),
    ("yellow", "pick up the yellow object", "object_4", "_11_banana"),
    ("eat", "pick up the thing you could eat", "object_4", "_11_banana"),
    ("negation", "pick up the object that is neither a toy nor a container", "object_4", "_11_banana"),
    ("puzzle", "pick up the puzzle toy", "object_3", "rubiks_cube"),
    ("colorful", "pick up the most colorful object", "object_3", "rubiks_cube"),
    ("nearbowl", "pick up the object closest to the bowl that is not red", "object_3", "rubiks_cube"),
    ("eatfrom", "pick up the object you would eat a meal from", "object_1", "_24_bowl"),
    ("between", "pick up the object between the cube and the banana", "object_1", "_24_bowl"),
    ("container", "pick up the round container", "object_1", "_24_bowl"),
]

PLACE_TASKS = [
    ("red", "put the block into the red box", "object_1", "red_bin"),
    ("green", "put the block into the green box", "object_0", "green_bin"),
    ("tomato", "put the block into the box that has the color of a tomato", "object_1", "red_bin"),
    ("grass", "put the block into the box that has the color of grass", "object_0", "green_bin"),
]

# every cluster of each pool -> the rigid object it grasps / the destination it lands on
PICK_CLUSTERS = {"object_0": "green_bin", "object_1": "_24_bowl", "object_2": "red_bin",
                 "object_3": "rubiks_cube", "object_4": "_11_banana"}
PLACE_CLUSTERS = {"object_0": "green_bin", "object_1": "red_bin", "object_2": "on rubiks_cube",
                  "object_3": "on _11_banana", "object_4": "on _24_bowl (rim)", "object_5": "on _24_bowl"}

LIFT_M = 0.15


def tasks(family):
    return PICK_TASKS if family == "pick" else PLACE_TASKS


def clusters(family):
    return PICK_CLUSTERS if family == "pick" else PLACE_CLUSTERS


def candidate_success(family, judge, target_cluster):
    """Did executing this candidate accomplish a task whose target is `target_cluster`?"""
    if family == "pick":
        obj = PICK_CLUSTERS[target_cluster]
        return bool(judge["lifted"][obj]["lifted"])
    bin_name = PLACE_CLUSTERS[target_cluster]
    return judge["place"] is not None and judge["place"]["landed_in"] == bin_name
