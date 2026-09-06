---
title: Mean over time
figure: Frames averaged into one
---

Averages every time point into a single frame, raising the signal-to-noise ratio by $\sqrt{T}$ for stationary structures and blurring anything that moves.

$$
M_{czyx} = \frac{1}{T}\sum_{t} I_{ctzyx}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Window** <br> all · running N | *All* averages the whole series into one frame; a running window of N frames keeps the time axis and averages each frame with its neighbours. |

## Note

Apply bleach correction before averaging a long series so early, brighter frames do not dominate.
