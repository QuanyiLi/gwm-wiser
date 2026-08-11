#!/usr/bin/env bash
# Shared task table for the scene-6 referring-expression eval (10 tasks).
# Sourced by run_refer6_tiptop.sh / run_refer6_gwm.sh.
#
# Rev2 (bins added to scene6 for the place eval) forced two rewordings, both
# user-approved 2026-08-11; results predating the bins are NOT comparable:
# - nearbowl: red_bin (0.176 m) is nearly as close to the bowl as the cube
#   (0.153 m), and "closest to the bowl" alone got fragile; "that is not red"
#   keeps the target the cube.
# - container: the table now has three containers (bowl + two bins);
#   "round container" keeps the target the bowl.
TAGS=(fruit yellow eat negation puzzle colorful nearbowl eatfrom between container)
declare -A INSTR RULE
INSTR[fruit]="pick up the fruit";                                            RULE[fruit]='{"objects":["banana"],"lift":0.15}'
INSTR[yellow]="pick up the yellow object";                                   RULE[yellow]='{"objects":["banana"],"lift":0.15}'
INSTR[eat]="pick up the thing you could eat";                                RULE[eat]='{"objects":["banana"],"lift":0.15}'
INSTR[negation]="pick up the object that is neither a toy nor a container";  RULE[negation]='{"objects":["banana"],"lift":0.15}'
INSTR[puzzle]="pick up the puzzle toy";                                      RULE[puzzle]='{"objects":["cube"],"lift":0.15}'
INSTR[colorful]="pick up the most colorful object";                          RULE[colorful]='{"objects":["cube"],"lift":0.15}'
INSTR[nearbowl]="pick up the object closest to the bowl that is not red";    RULE[nearbowl]='{"objects":["cube"],"lift":0.15}'
INSTR[eatfrom]="pick up the object you would eat a meal from";               RULE[eatfrom]='{"objects":["bowl"],"lift":0.15}'
INSTR[between]="pick up the object between the cube and the banana";         RULE[between]='{"objects":["bowl"],"lift":0.15}'
INSTR[container]="pick up the round container";                              RULE[container]='{"objects":["bowl"],"lift":0.15}'
