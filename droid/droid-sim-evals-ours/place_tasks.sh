#!/usr/bin/env bash
# Shared task table for the scene-6 place eval (4 tasks, scene 6 variant 1).
# Sourced by the place runners. The referring expression is on the DESTINATION
# (the object in hand is fixed and welded); expressions split 2-2 between the
# bins so a pure side-preference policy scores at most 2/4 -- without the split
# a "reach left always" policy is indistinguishable from colour grounding.
#
# Success: the block's mesh centre inside the named bin's volume at episode end
# (still gripped -- plans end at the placement pose, no release, no go-home).
# xy_tol 0.05 < bin half-width 0.0575; bins are 0.215 m apart so a wrong-bin
# landing can never satisfy the right bin. "candidates" makes PlaceTracker
# record which bin actually got the block (detail._landed_in) so wrong-bin vs
# drop/plan-failure separate cleanly in the CSV.
TAGS=(red green tomato grass)
declare -A INSTR RULE
_R() { printf '{"objects":["held_block"],"container":"%s","candidates":["red_bin","green_bin"],"xy_tol":0.05,"z_rel":[-0.03,0.03]}' "$1"; }
INSTR[red]="put the block into the red box";                                RULE[red]=$(_R red_bin)
INSTR[green]="put the block into the green box";                            RULE[green]=$(_R green_bin)
INSTR[tomato]="put the block into the box that has the color of a tomato";  RULE[tomato]=$(_R red_bin)
INSTR[grass]="put the block into the box that has the color of grass";      RULE[grass]=$(_R green_bin)
