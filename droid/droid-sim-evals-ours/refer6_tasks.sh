#!/usr/bin/env bash
# Shared task table for the scene-6 referring-expression eval (10 tasks):
# instructions and success rules for the scene-6 referring tasks.
# Sourced by run_refer6_tiptop.sh / run_refer6_gwm.sh.
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
