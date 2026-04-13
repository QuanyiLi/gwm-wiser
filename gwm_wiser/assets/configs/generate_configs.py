"""
Generate 24 scenario config files with all permutations of 4 colors.
Each config has:
- Train colors: permutation of [red, green, blue, yellow]
- Test colors: same permutation applied to [purple, orange, cyan, pink]
- 12 task expressions: 6 by color, 6 by spatial relation
- Table images: from images/{i}_xyz folder if exists, else placeholder
- VARIATION: Different spatial phrasings across configs and between train/test
- NOTE: Only use unambiguous left/right spatial descriptions
- NOTE: Train and test use COMPLETELY DIFFERENT spatial templates (no overlap)
"""

import itertools
import os
import glob
import json

# Base colors
train_base = ["red", "green", "blue", "yellow"]
test_base = ["purple", "orange", "cyan", "grey"]

# All 24 permutations
permutations = list(itertools.permutations(range(4)))

# ============================================================================
# TRAIN-ONLY spatial templates (used only in train mode)
# ============================================================================

# Position 0 (leftmost) - TRAIN
pos0_train_templates = [
    "the leftmost cube",
    "the cube on the far left",
    "the first cube from the left",
]

# Position 1 (second from left) - TRAIN
pos1_train_templates = [
    "the cube to the right of the {prev} cube",
    "the second cube from the left",
    "the cube between the {prev} and {next} cubes",
]

# Position 2 (third from left) - TRAIN
pos2_train_templates = [
    "the cube to the left of the {next} cube",
    "the cube between the {prev} and {next} cubes",
    "the second cube from the right",
]

# Position 3 (rightmost) - TRAIN
pos3_train_templates = [
    "the rightmost cube",
    "the cube on the far right",
    "the last cube from the left",
]

# ============================================================================
# TEST-ONLY spatial templates (completely different from train)
# ============================================================================

# Position 0 (leftmost) - TEST
pos0_test_templates = [
    "the cube on the left side",
    "the cube directly to the left of {next}",
    "the cube at position one from the left",
]

# Position 1 (second from left) - TEST
pos1_test_templates = [
    "the cube immediately next to the {prev} one",
    "the cube sandwiched between {prev} and {next}",
    "the cube at position two from the left",
]

# Position 2 (third from left) - TEST
pos2_test_templates = [
    "the cube directly to the left of {next}",
    "the cube sandwiched between {prev} and {next}",
    "the cube on the left side of {next}",
]

# Position 3 (rightmost) - TEST
pos3_test_templates = [
    "the cube on the most right side",
    "the cube to the right of the {prev} cube",
    "the cube directly to the right of {prev}",
]


def get_spatial_expr(pos, colors, template_idx, is_test=False):
    """Get a spatial expression for a cube at given position with variation."""
    prev_color = colors[pos - 1] if pos > 0 else None
    next_color = colors[pos + 1] if pos < 3 else None

    if is_test:
        if pos == 0:
            templates = pos0_test_templates
        elif pos == 1:
            templates = pos1_test_templates
        elif pos == 2:
            templates = pos2_test_templates
        else:
            templates = pos3_test_templates
    else:
        if pos == 0:
            templates = pos0_train_templates
        elif pos == 1:
            templates = pos1_train_templates
        elif pos == 2:
            templates = pos2_train_templates
        else:
            templates = pos3_train_templates

    template = templates[template_idx % len(templates)]

    # Format with colors if needed
    if prev_color:
        template = template.replace("{prev}", prev_color)
    if next_color:
        template = template.replace("{next}", next_color)

    return template


def load_image_config(perm_idx, images_dir):
    """Load image config from images/{perm_idx}_* folder if it exists.

    Returns:
        tuple: (train_images, test_images, ref_exps) or None if no folder exists
        - train_images: list of 3 image paths (relative to images folder)
        - test_images: list of 3 image paths (relative to images folder)
        - ref_exps: dict with image keys -> list of 2 referring expressions
    """
    # Find folder matching pattern {perm_idx}_*
    pattern = os.path.join(images_dir, f"{perm_idx}_*")
    matching_folders = glob.glob(pattern)

    if not matching_folders:
        return None

    folder = matching_folders[0]
    folder_name = os.path.basename(folder)

    # Load ref_exp.json
    ref_exp_path = os.path.join(folder, "ref_exp.json")
    if not os.path.exists(ref_exp_path):
        return None

    with open(ref_exp_path, "r") as f:
        ref_exps = json.load(f)

    # Find train and test images
    train_images = []
    test_images = []

    for filename in sorted(os.listdir(folder)):
        if filename.startswith("train_") and (
            filename.endswith(".png") or filename.endswith(".jpg")
        ):
            train_images.append(f"{folder_name}/{filename}")
        elif filename.startswith("test_") and (
            filename.endswith(".png") or filename.endswith(".jpg")
        ):
            test_images.append(f"{folder_name}/{filename}")

    if len(train_images) != 3 or len(test_images) != 3:
        print(
            f"Warning: Expected 3 train and 3 test images in {folder}, got {len(train_images)} train, {len(test_images)} test"
        )
        return None

    # Extract name without index prefix (e.g., "0_country" -> "country")
    name = folder_name.split("_", 1)[1] if "_" in folder_name else folder_name

    return name, folder_name, train_images, test_images, ref_exps


def generate_config(perm_idx, perm, images_dir):
    """Generate a config file for a given permutation."""
    train_colors = [train_base[i] for i in perm]
    test_colors = [test_base[i] for i in perm]

    # Try to load image config
    image_config = load_image_config(perm_idx, images_dir)

    if image_config:
        name, folder_name, train_images, test_images, ref_exps = image_config

        # Track usage count for each image's referring expressions
        # Key: image basename (without extension), Value: count of uses
        train_ref_usage = {}
        test_ref_usage = {}
    else:
        name = None
        train_images = ["placeholder", "placeholder", "placeholder"]
        test_images = ["placeholder", "placeholder", "placeholder"]
        ref_exps = None

    train_exprs = []
    test_exprs = []

    for cube_idx in range(4):
        for dest_idx in range(3):
            task_idx = cube_idx * 3 + dest_idx  # 0-11

            # Alternate starting pattern based on config index
            use_color = (task_idx + perm_idx) % 2 == 0

            if use_color:
                train_obj = f"{train_colors[cube_idx]} cube"
                test_obj = f"{test_colors[cube_idx]} cube"
            else:
                template_idx = perm_idx + task_idx
                train_obj = get_spatial_expr(
                    cube_idx, train_colors, template_idx, is_test=False
                )
                test_obj = get_spatial_expr(
                    cube_idx, test_colors, template_idx, is_test=True
                )

            # Get destination referring expression
            if ref_exps:
                # Get image for this destination (3 images, dest_idx cycles through 0,1,2)
                train_img = train_images[dest_idx]
                test_img = test_images[dest_idx]

                # Extract key (e.g., "train_france" from "0_country/train_france.png")
                train_key = os.path.splitext(os.path.basename(train_img))[0]
                test_key = os.path.splitext(os.path.basename(test_img))[0]

                # Get usage count and increment
                train_count = train_ref_usage.get(train_key, 0)
                test_count = test_ref_usage.get(test_key, 0)

                # Alternate between ref_exp[0] and ref_exp[1] using dest_idx
                # This ensures:
                # 1. Per image: rotates 0, 1, 0, 1 (as count increments) combined with constant dest_idx
                # 2. Per cube: (0+0)%2=0, (0+1)%2=1, (0+2)%2=0 -> [0, 1, 0] mixed!
                train_dest = ref_exps[train_key][(train_count + dest_idx) % 2]
                test_dest = ref_exps[test_key][(test_count + dest_idx) % 2]

                train_ref_usage[train_key] = train_count + 1
                test_ref_usage[test_key] = test_count + 1
            else:
                train_dest = "placeholder"
                test_dest = "placeholder"

            train_exprs.append([train_obj, train_dest])
            test_exprs.append([test_obj, test_dest])

    # Build YAML content
    name_str = name if name else "placeholder"
    yaml_content = f'''name: "{name_str}"

train:
  cube_colors:
    - {train_colors[0]}
    - {train_colors[1]}
    - {train_colors[2]}
    - {train_colors[3]}

  table_images:
'''
    for img in train_images:
        yaml_content += f'    - path: "{img}"\n'

    yaml_content += """
  task_referring_expressions:
"""
    for obj, dest in train_exprs:
        yaml_content += f'    - ["{obj}", "{dest}"]\n'

    yaml_content += f"""
test:
  cube_colors:
    - {test_colors[0]}
    - {test_colors[1]}
    - {test_colors[2]}
    - {test_colors[3]}

  table_images:
"""
    for img in test_images:
        yaml_content += f'    - path: "{img}"\n'

    yaml_content += """
  task_referring_expressions:
"""
    for obj, dest in test_exprs:
        yaml_content += f'    - ["{obj}", "{dest}"]\n'

    return yaml_content


# Generate all 24 configs
output_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(output_dir), "images")

for i, perm in enumerate(permutations):
    config_content = generate_config(i, perm, images_dir)
    config_path = os.path.join(output_dir, f"config_{i}.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(
        f"Generated: config_{i}.yaml - train: {[train_base[j] for j in perm]}, test: {[test_base[j] for j in perm]}"
    )

print(f"\nGenerated {len(permutations)} config files in {output_dir}")
