---
title: Einstein-summation reduction
figure: Axis diagram: kept vs. reduced
---

Reduces the array along any set of axes with one operation — the expression lists input axes and the axes that survive. Removed axes are collapsed with the chosen reduction.

$$
\text{ctzyx} \rightarrow \text{czyx}:\quad R_{czyx} = \frac{1}{T}\sum_{t} I_{ctzyx}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Axes** <br> keep · reduce | Click an axis to toggle whether it is kept in the output or reduced away. Reduced axes remain in the array with length 1, so downstream steps never have to guess the axis order. |
| **Reduction** <br> sum · mean · max · min | Applied to every reduced axis. Max over Z is a maximum-intensity projection; mean over T averages a time series. |

## Note

The expression `ctzyx -> czyx` is the same notation NumPy's `einsum` uses; the exported Python script reproduces the step with it.
