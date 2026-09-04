#include "qt/help_text.hpp"

namespace sirius::app {

    // Qt's rich text engine has no math renderer, so the formulas below use
    // Unicode and <sub>/<sup>; each is set in its own indented block.
    QString helpHtml() {
        return QStringLiteral(R"(
<style>
  body { font-family: sans-serif; }
  h2 { margin-top: 18px; }
  .eq { font-family: 'DejaVu Serif', serif; font-size: 105%; margin: 6px 0 6px 28px; }
  .note { color: #555; }
  table { border-collapse: collapse; }
  td { padding: 3px 10px 3px 0; vertical-align: top; }
</style>

<a name="overview"></a><h2>What structured illumination reconstruction does</h2>
<p>A widefield microscope transmits object frequencies only inside the OTF support,
a disc of radius <b>2NA/λ</b> (the white circle on the spectra). Structured
illumination multiplies the sample by a sinusoidal pattern, which in Fourier
space shifts copies of the object spectrum by multiples of the pattern vector
<b>k</b><sub>0</sub>, so frequencies from outside the support are folded into it
and recorded. The reconstruction unmixes those copies, moves them back to where
they belong, and combines them into a spectrum of extended support.</p>

<p>For three-beam (3D) SIM the illumination along a direction <i>d</i> with
phase <i>φ</i> is a sum of harmonic orders <i>o</i> = 0, ±1, ±2 (two-beam 2D SIM
has 0, ±1):</p>
<p class="eq">I(<b>r</b>) = Σ<sub>o</sub> a<sub>o</sub> · e<sup> i o (2π <b>k</b><sub>0</sub>·<b>r</b> + φ)</sup></p>
<p>so every raw frame spectrum is a superposition of shifted object spectra
<i>Õ</i>, each seen through the OTF <i>H</i> of that order:</p>
<p class="eq">D̃<sub>d,φ</sub>(<b>k</b>) = Σ<sub>o</sub> a<sub>o</sub> e<sup> i o φ</sup> · H<sub>o</sub>(<b>k</b>) · Õ(<b>k</b> − o <b>k</b><sub>0,d</sub>)</p>
<p>In 3D the order ±1 components also carry an axial modulation, which is why
the order-1 OTF (see the OTF tab) is the widefield OTF shifted by ±k<sub>z</sub>
of the first illumination order.</p>

<a name="pipeline"></a><h2>Pipeline</h2>
<ol>
<li><b>Preprocessing.</b> The camera <i>background</i> is subtracted, frames are
scaled by a global factor, <i>bleach correction</i> equalizes the total intensity
of every frame to direction 0 / phase 0, and the <i>input apodization</i>
softens the edges so the FFT does not see a hard border.</li>
<li><b>Band separation.</b> With <i>n</i><sub>φ</sub> ≥ 2·orders − 1 phase steps the
frames of one direction are a linear system in the bands: the widefield band
(order 0) and the cosine and sine parts of every order (the Bands tab shows
them as the ±<i>o</i> side bands re ± i·im). The separation matrix is the
inverse of the phase matrix; for equally spaced phases
φ<sub>j</sub> = 2πj/n<sub>φ</sub> it is the analytic cos / sin matrix.</li>
<li><b>Pattern vector.</b> Starting from the <i>line spacing</i> and the
<i>k0 start angle</i> (yellow crosses on the raw spectrum), the order-0 and
order-1 (2D) or order-2 (3D) bands are whitened by each other's OTF in the
region where they overlap and cross-correlated; the peak gives
<b>k</b><sub>0</sub> to sub-pixel precision. A bracket search over angle and
magnitude then maximizes the modulation, which refines <b>k</b><sub>0</sub>
(cyan circles after a run).</li>
<li><b>Modulation amplitudes.</b> For each order the complex amplitude
<i>a</i><sub>o</sub> relating the shifted band to band 0 is a least-squares fit
over the overlap:</li>
</ol>
<p class="eq">a<sub>o</sub> = Σ conj(B̃<sub>0</sub>) · B̃<sub>o</sub>(<b>k</b> + o<b>k</b><sub>0</sub>) / Σ |B̃<sub>0</sub>|²</p>
<ol start="5">
<li><b>Generalized Wiener filter and assembly.</b> Every band is shifted to
its true position on an output grid enlarged by the <i>zoom factors</i> and the
bands are combined with weights that favour the order with the stronger OTF at
each frequency:</li>
</ol>
<p class="eq">Õ(<b>k</b>) = A(<b>k</b>) · Σ<sub>d,o</sub> conj(a<sub>o</sub> H<sub>o</sub>(<b>k</b> − o<b>k</b><sub>0,d</sub>)) · B̃<sub>d,o</sub>(<b>k</b> − o<b>k</b><sub>0,d</sub>)
&nbsp;/&nbsp; ( Σ<sub>d,o</sub> |a<sub>o</sub> H<sub>o</sub>(<b>k</b> − o<b>k</b><sub>0,d</sub>)|² + w² )</p>
<p>where <i>w</i> is the <i>Wiener constant</i> and <i>A</i> the <i>output
apodization</i>. The inverse FFT of <i>Õ</i> is the result. The Bands tab's
"Wiener filtered" stage shows the bands right before they are moved and summed.</p>

<a name="optics"></a><h2>Illumination and optics</h2>
<table>
<tr><td><b>Directions, Phases, Orders</b></td><td>Geometry of the acquisition:
the stack holds directions × phases × z sections in direction → z → phase
order. Orders is the number of harmonics to separate (3 for three-beam 3D
SIM, 2 for two-beam 2D SIM) and needs phases ≥ 2·orders − 1.</td></tr>
<tr><td><b>k0 start angle</b></td><td>Orientation of direction 0's pattern; the
other directions follow at +π/N steps unless the file lists explicit angles.
Only a starting point for the search.</td></tr>
<tr><td><b>Line spacing</b></td><td>Period <i>p</i> of the finest pattern in the
sample plane; the starting |<b>k</b><sub>0</sub>| is 1/<i>p</i> (2D) or
1/(2<i>p</i>) for order 1 in 3D. Check it against the yellow crosses on the raw
spectrum: they should sit on the illumination peaks.</td></tr>
<tr><td><b>NA, immersion index, wavelength</b></td><td>Define the OTF support
2NA/λ and the axial cutoff (n − √(n² − NA²))/λ, and the ideal OTF when no OTF
file is loaded. The excitation wavelength is approximated as 0.88 λ.</td></tr>
</table>

<a name="sampling"></a><h2>Sampling</h2>
<table>
<tr><td><b>dx, dy, dz</b></td><td>Voxel size of the raw stack. A spectrum of N
pixels has a frequency step of 1/(N·d), which is what the overlay geometry and
the readout use.</td></tr>
<tr><td><b>dz of PSF/OTF</b></td><td>z step of the bead stack behind a measured
OTF (or of the simulated PSF of the ideal one); it fixes the OTF's axial
frequency step.</td></tr>
<tr><td><b>Lateral / axial zoom</b></td><td>Size of the output grid relative to
the input. The extended support reaches 2NA/λ + 2|<b>k</b><sub>0</sub>|, about
twice the widefield limit, so a lateral zoom of 2 keeps it alias-free.</td></tr>
</table>

<a name="filtering"></a><h2>Filtering</h2>
<table>
<tr><td><b>Wiener constant</b></td><td><i>w</i> in the filter above: a floor on the
denominator that stops frequencies with a weak OTF from being amplified into
noise. Larger = smoother, smaller = sharper and noisier.</td></tr>
<tr><td><b>OTF cutoff</b></td><td>Fraction of the OTF peak below which a frequency
is excluded from the overlap regions used by the pattern-vector and amplitude
fits (order 0 is allowed 5× more in 3D).</td></tr>
<tr><td><b>Background</b></td><td>Camera offset subtracted first. It biases the
bleach correction and the modulation amplitudes when wrong.</td></tr>
<tr><td><b>Input apodization, border width</b></td><td>Triangle blends opposite
image edges over the border width so the periodic FFT sees no step; Cosine
applies a sine window; None leaves the frames as they are.</td></tr>
<tr><td><b>Output apodization</b></td><td>Taper <i>A</i>(<b>k</b>) towards the edge
of the extended support, against ringing in the result.</td></tr>
<tr><td><b>Suppress singularities, suppression radius</b></td><td>Notch around
the band centers of orders ≥ 1 where the residual pattern peak would otherwise
print as stripes.</td></tr>
<tr><td><b>Dampen order 0</b></td><td>Lower the widefield band's weight near the
origin to reduce out-of-focus haze.</td></tr>
<tr><td><b>Bleach correction, equalize across z</b></td><td>Scale frames to a
common total intensity, per z plane or (equalize) to plane 0 of direction 0.</td></tr>
<tr><td><b>Skip kz = 0</b></td><td>3D: leave the unreliable kz = 0 plane (missing
cone) out of the fits and the order-0 weight.</td></tr>
<tr><td><b>Filter overlaps</b></td><td>Use the Wiener cross-weights where bands
overlap; off sums them plainly.</td></tr>
</table>

<a name="run"></a><h2>Running</h2>
<p><b>OTF.</b> A radially averaged OTF TIFF measured from beads gives the best
result. Without one, the theoretical OTF of an aberration-free objective is
computed from NA, immersion index and wavelength (3D with the missing cone
when the stack has several planes). Modulation amplitudes are measured relative
to the OTF, so their values differ between the two.</p>
<p><b>Device</b> selects the CPU (FFTW) or a CUDA GPU (cuFFT) backend; the
numerics are identical. <b>FFT planning</b> is FFTW's planner effort: Measure is
a good default, Estimate plans instantly but runs slower, Patient plans for a
long time. Plans are kept between runs with the same parameters.</p>
<p><b>Capture intermediate spectra</b> keeps the separated and Wiener-filtered
bands for the Bands tab. Off by default: two copies of every band volume.</p>

<a name="viewer"></a><h2>Viewers</h2>
<p>Mouse wheel zooms around the cursor, left drag pans (or draws a selection
when <b>Select</b> is on), <b>Fit</b> and <b>1:1</b> reset the zoom. <b>Min</b> /
<b>Max</b> set the display window; <b>Auto</b> uses percentiles, <b>Reset</b> the
full range; <b>Log</b> shows log<sub>10</sub>. <b>Crop</b> opens the selection
through every slice in a new tab. <b>Ortho</b> adds XZ and YZ sections through a
crosshair you place by clicking (navigation is locked meanwhile); <b>Physical
z</b> gives z its true proportion dz/dx. <b>Spectrum</b> shows the centered
|FFT| of each displayed plane with the <b>Overlay</b>: white circle = OTF
support, yellow + = illumination peaks expected from the parameters, cyan
circles = fitted pattern vectors with |a|. Double-click a cell of the Bands
grid to open it in a full viewer.</p>
)");
    }

} // namespace sirius::app
