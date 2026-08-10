---
status: accepted
---

# Preserve RAT in the Qwen visual latent

The robot-video model will preserve the original rendering-based action tokenization scheme: current full RGB plus future robot-only RGB conditions prediction of the future full-RGB trajectory in frozen Qwen3-VL internal visual tokens. Raw-action conditioning and direct prediction of the pooled retrieval vector were rejected because they change the successful GWM interface and discard the visual trajectory representation central to the original result.

## Consequences

A candidate trajectory must provide future robot appearances even though GWM training does not require actions, proprioception, task text, or captions. Language enters later when the external scorer compares a task query with the predicted trajectory embedding; it is not a GWM forward input. The frozen-Qwen, MSE, and cosine settings remain recorded in the phase plan.
