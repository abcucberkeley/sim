---
title: Deskew and rotate
figure: Sheared vs. deskewed stack
---

Light-sheet stacks are acquired at an angle to the coverslip: the stage moves the sample between planes, so each plane is displaced along x relative to the previous one. Deskewing shears each plane by the stage step and rotates the volume so Z is normal to the coverslip.

$$
\Delta x = \Delta z_{\text{stage}} \cdot \cos\theta
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Sheet angle θ** <br> 30 – 35° | Angle between the light sheet and the coverslip; from the microscope's calibration. |
| **Stage step** <br> µm | Distance the stage moves between planes. The shear per plane in pixels is $\Delta z_{\text{stage}} \cos\theta / dx$. |
| **Rotate to coverslip** <br> on · off | After shearing, rotate by θ so the output z axis is normal to the coverslip and resample onto an isotropic grid of the lateral pixel size. |
| **Interpolation** <br> linear · cubic | Cubic preserves detail; linear is faster. Nearest is only useful for label volumes. |

## Note

The step validates that the dataset is light-sheet data (from the Load step's acquisition metadata) and warns otherwise; the zero fill outside the sheared volume is what the stitching step masks as background.
