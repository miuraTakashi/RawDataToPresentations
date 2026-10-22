#!/usr/bin/env python3
"""FRAP figures for Keynote, following visualize_frap.py.

Stimulation ROIs are circles stored in the ND2. A square of twice the ROI
diameter is cropped, displayed with one intensity range for every frame, and
the mean intensity inside the ROI is plotted against time. The slide figure
is four representative frames above that trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from PIL import Image

DISPLAY_LO = 0.5
DISPLAY_HI = 99.5


@dataclass
class CircleROI:
    roi_id: int
    cx: float
    cy: float
    diameter: float

    @property
    def radius(self) -> float:
        return self.diameter / 2


@dataclass
class Timing:
    time_s: np.ndarray
    stim_begin: float | None
    stim_end: float | None
    shade_begin: float | None
    shade_end: float | None
    n_pulses: int


def configure_font() -> None:
    root = Path("/System/Library/Fonts")
    if not root.is_dir():
        return
    hits = [p for p in root.iterdir() if p.suffix == ".ttc" and p.name.endswith("W3.ttc")]
    if not hits:
        return
    font_manager.fontManager.addfont(str(hits[0]))
    name = font_manager.FontProperties(fname=str(hits[0])).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False


def time_stack(nd) -> np.ndarray:
    """Return fluorescence as (T, Y, X), taking the first index of other axes."""
    array = np.asarray(nd.asarray(), dtype=np.float32)
    axes = [str(name) for name in nd.sizes]
    if len(axes) != array.ndim or tuple(int(nd.sizes[name]) for name in axes) != array.shape:
        while array.ndim > 3:
            array = array[0]
        if array.ndim == 2:
            array = array[None, ...]
        return array

    index = []
    kept = []
    for axis in axes:
        if axis in ("T", "Y", "X"):
            index.append(slice(None))
            kept.append(axis)
        else:
            index.append(0)
    stack = np.asarray(array[tuple(index)])
    if "T" not in kept:
        stack = stack[None, ...]
        kept = ["T", *kept]
    order = [kept.index(name) for name in ("T", "Y", "X")]
    return np.transpose(stack, order)


def stimulation_circles(nd) -> list[CircleROI]:
    """Stimulation ROIs in pixel coordinates. Origin is the top left, y points down."""
    width = int(nd.sizes["X"])
    height = int(nd.sizes["Y"])
    circles: list[CircleROI] = []
    for roi in nd.rois.values():
        if roi.info.interpType.name != "StimulationROI":
            continue
        if not roi.animParams:
            continue
        anim = roi.animParams[0]
        cx = (anim.centerX + 1) / 2 * width
        cy = (anim.centerY + 1) / 2 * height
        roi_w = anim.boxShape.sizeX / 2 * width
        roi_h = anim.boxShape.sizeY / 2 * height
        circles.append(CircleROI(roi.id, cx, cy, max(roi_w, roi_h)))
    return circles


def frame_timing(nd, n_frames: int) -> Timing:
    events = list(nd.events())
    frames = [event for event in events if "T Index" in event and "Time [s]" in event]
    frames.sort(key=lambda event: int(event["T Index"]))
    if len(frames) >= n_frames:
        time_s = np.array([float(event["Time [s]"]) for event in frames[:n_frames]], dtype=float)
    else:
        time_s = np.arange(n_frames, dtype=float)

    def event_times(name: str) -> list[float]:
        return [float(event["Time [s]"]) for event in events if event.get("Events") == name]

    begins = event_times("Stimulation Phase Begin")
    ends = event_times("Stimulation Phase End")
    hv_off = event_times("StandardPMT(All Ch)HV=0")
    hv_on = event_times("StandardPMT(All Ch)HV restored")
    n_pulses = sum(1 for event in events if event.get("Events") == "Stimulation Phase Sequential")

    stim_begin = min(begins) if begins else None
    stim_end = max(ends) if ends else None
    shade_begin, shade_end = stim_begin, stim_end
    if stim_begin is not None and stim_end is not None and stim_end - stim_begin < 0.05:
        if hv_off and hv_on:
            shade_begin, shade_end = min(hv_off), max(hv_on)
    return Timing(time_s, stim_begin, stim_end, shade_begin, shade_end, n_pulses)


def square_window(cx: float, cy: float, diameter: float, width: int, height: int):
    side = max(1, int(round(diameter * 2)))
    col0 = int(round(cx - side / 2))
    row0 = int(round(cy - side / 2))
    return col0, row0, side


def crop_square(stack: np.ndarray, col0: int, row0: int, side: int):
    n, height, width = stack.shape
    crop = np.zeros((n, side, side), dtype=np.float32)
    valid = np.zeros((side, side), dtype=bool)
    src_c0, src_r0 = max(col0, 0), max(row0, 0)
    src_c1, src_r1 = min(col0 + side, width), min(row0 + side, height)
    if src_c1 <= src_c0 or src_r1 <= src_r0:
        return crop, valid
    dst_c0, dst_r0 = src_c0 - col0, src_r0 - row0
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    crop[:, dst_r0:dst_r1, dst_c0:dst_c1] = stack[:, src_r0:src_r1, src_c0:src_c1]
    valid[dst_r0:dst_r1, dst_c0:dst_c1] = True
    return crop, valid


def circle_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    cols = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    dist = np.hypot(cols[None, :] - cx, rows[:, None] - cy)
    return dist <= radius


def display_limits(crop: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    values = crop[:, valid]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [DISPLAY_LO, DISPLAY_HI])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def measure(stack: np.ndarray, mask: np.ndarray, timing: Timing) -> dict:
    means = stack[:, mask].mean(axis=1).astype(float)
    n_pixels = int(mask.sum())
    t = timing.time_s
    if timing.stim_begin is None:
        pre = np.zeros(len(t), dtype=bool)
        post = np.ones(len(t), dtype=bool)
    else:
        end = timing.stim_begin if timing.stim_end is None else timing.stim_end
        pre = t < timing.stim_begin
        post = t > end
    i0 = float(means[pre][-1]) if np.any(pre) else float(means[0])
    i_post = float(means[post][0]) if np.any(post) else float("nan")
    i_end = float(means[post][-1]) if np.any(post) else float(means[-1])
    return {
        "mean": means,
        "n_pixels": n_pixels,
        "pre": pre,
        "post": post,
        "i0": i0,
        "i_post": i_post,
        "i_end": i_end,
    }


def channel_color_name(meta: dict) -> str:
    """Map the acquisition channel to red, green, or blue."""
    name = str(meta.get("channel") or "").lower()
    if any(token in name for token in ("dapi", "hoechst", "405")):
        return "blue"
    if any(token in name for token in ("488", "gfp", "egfp", "fitc", "alexa488", "alexa 488")):
        return "green"
    if any(
        token in name
        for token in (
            "568", "555", "594", "561", "cy3", "cy5", "mcherry", "m-cherry",
            "tdtomato", "tritc", "texas red", "rhodamine", "alexa568", "alexa 568",
        )
    ):
        return "red"
    wave = meta.get("wavelength_nm")
    if isinstance(wave, (int, float)) and wave > 0:
        if wave < 500:
            return "blue"
        if wave < 560:
            return "green"
        return "red"
    return "green"


def channel_rgb(meta: dict) -> tuple[float, float, float]:
    return {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }[channel_color_name(meta)]


def fluorescence_cmap(rgb: tuple[float, float, float]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("channel", [(0.0, 0.0, 0.0), rgb])


def draw_trace(ax, timing: Timing, measured: dict, color: str | tuple[float, float, float] = "black") -> None:
    t = timing.time_s
    y = measured["mean"]
    labeled = False
    for mask in (measured["pre"], measured["post"]):
        if not np.any(mask):
            continue
        ax.plot(
            t[mask],
            y[mask],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label="ROI" if not labeled else None,
        )
        labeled = True
    if timing.shade_begin is not None and timing.shade_end is not None:
        span = float(t[-1] - t[0]) if len(t) else 1.0
        width = timing.shade_end - timing.shade_begin
        if width < 0.02 * max(span, 1e-6):
            ax.axvline(
                (timing.shade_begin + timing.shade_end) / 2,
                color="#d95f5f",
                linewidth=1.2,
                label="Bleach",
                zorder=0,
            )
        else:
            ax.axvspan(
                timing.shade_begin,
                timing.shade_end,
                color="#d95f5f",
                alpha=0.28,
                label="Bleach",
                zorder=0,
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean intensity")
    ax.set_ylim(0, max(float(np.max(y)) * 1.12, 1))
    ax.set_xlim(min(0.0, float(t[0])), float(t[-1]) * 1.02)
    ax.yaxis.grid(True, color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        borderaxespad=0.0,
        fontsize=10,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def representative_indices(measured: dict) -> list[tuple[int, str]]:
    pre = np.flatnonzero(measured["pre"])
    post = np.flatnonzero(measured["post"])
    pre_i = int(pre[-1]) if len(pre) else 0
    post_i = int(post[0]) if len(post) else min(1, len(measured["mean"]) - 1)
    end_i = int(post[-1]) if len(post) else len(measured["mean"]) - 1
    mid_i = post_i + (end_i - post_i) // 2
    return [(pre_i, "Pre-bleach"), (post_i, "Post-bleach"), (mid_i, "Mid"), (end_i, "End")]


def pad_square(path: Path, margin_ratio: float = 0.04) -> None:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = max(width, height)
    margin = int(round(side * margin_ratio))
    canvas_side = side + 2 * margin
    canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))
    canvas.paste(image, ((canvas_side - width) // 2, (canvas_side - height) // 2))
    canvas.save(path)


def write_panel(
    path: Path,
    crop: np.ndarray,
    col0: int,
    row0: int,
    roi: CircleROI,
    vmin: float,
    vmax: float,
    timing: Timing,
    measured: dict,
    channel_rgb_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    fig = plt.figure(figsize=(7.4, 7.4), dpi=160)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.9],
        left=0.12,
        right=0.97,
        top=0.96,
        bottom=0.08,
        hspace=0.48,
        wspace=0.10,
    )
    side = crop.shape[1]
    cx_c = roi.cx - col0
    cy_c = roi.cy - row0
    for column, (index, label) in enumerate(representative_indices(measured)):
        ax = fig.add_subplot(gs[0, column])
        ax.imshow(
            crop[index],
            cmap=fluorescence_cmap(channel_rgb_color),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(0, side, side, 0),
        )
        ax.add_patch(
            Circle(
                (cx_c, cy_c),
                roi.radius,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)
        ax.set_title(f"{label}\n{timing.time_s[index]:.1f} s", fontsize=11, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
            spine.set_linewidth(0.6)
    ax = fig.add_subplot(gs[1, :])
    draw_trace(ax, timing, measured, color=channel_rgb_color)
    fig.savefig(path)
    plt.close(fig)
    pad_square(path, margin_ratio=0.04)


def acquisition_text(nd) -> dict:
    text = nd.text_info or {}
    description = str(text.get("description", "")) + "\n" + str(text.get("capturing", ""))
    power = re.search(r"Laser Power\}\s*:\s*([0-9.]+)", description)
    hv = re.search(r"PMT HV\}\s*:\s*([0-9.]+)", description)
    wave = re.search(r"Laser Wavelength\}\s*:\s*([0-9.]+)", description)
    channel = nd.metadata.channels[0]
    return {
        "channel": channel.channel.name,
        "wavelength_nm": float(wave.group(1)) if wave else channel.channel.excitationLambdaNm,
        "laser_power_percent": float(power.group(1)) if power else None,
        "pmt_hv": float(hv.group(1)) if hv else None,
        "zoom": float(channel.microscope.zoomMagnification),
        "objective": channel.microscope.objectiveName,
        "um_per_px": float(nd.voxel_size().x),
    }


def ratio(num: float, den: float) -> float | None:
    if den == 0 or not np.isfinite(num) or not np.isfinite(den):
        return None
    return float(num / den)


def bullet_lines(meta: dict, roi: CircleROI, timing: Timing, measured: dict) -> list[str]:
    power = meta["laser_power_percent"]
    hv = meta["pmt_hv"]
    wave = meta["wavelength_nm"]
    optical = f"{wave:.0f} nm"
    if power is not None:
        optical += f"、{power:.0f}%"
    if hv is not None:
        optical += f"、HV {hv:.0f}"
    zoom = meta["zoom"]
    zoom_text = f"{zoom:.0f}" if zoom == int(zoom) else f"{zoom:g}"
    diameter_um = roi.diameter * meta["um_per_px"]
    post_ratio = ratio(measured["i_post"], measured["i0"])
    end_ratio = ratio(measured["i_end"], measured["i0"])
    recovery = "輝度比を算出できない"
    if post_ratio is not None and end_ratio is not None:
        recovery = f"bleach前比 {post_ratio:.2f} から {end_ratio:.2f}"
    return [
        optical,
        f"ズーム {zoom_text}、{meta['um_per_px']:.3f} μm/px",
        f"ROI 直径 {diameter_um:.2f} μm",
        f"退色 {timing.n_pulses} 回",
        recovery,
    ]


def analyze_nd2(path: Path, panel_path: Path) -> list[dict]:
    """Build one panel per stimulation ROI. Returns slide records, or [] if none."""
    import nd2

    configure_font()
    records = []
    with nd2.ND2File(path) as ndfile:
        circles = stimulation_circles(ndfile)
        if not circles:
            print(f"  skip {path.name}: 刺激 ROI がない")
            return records
        stack = time_stack(ndfile)
        timing = frame_timing(ndfile, stack.shape[0])
        meta = acquisition_text(ndfile)
        height, width = stack.shape[1], stack.shape[2]
        for roi in circles:
            suffix = "" if len(circles) == 1 else f"_roi{roi.roi_id}"
            dest = panel_path if len(circles) == 1 else panel_path.with_name(f"{panel_path.stem}{suffix}{panel_path.suffix}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            col0, row0, side = square_window(roi.cx, roi.cy, roi.diameter, width, height)
            crop, valid = crop_square(stack, col0, row0, side)
            vmin, vmax = display_limits(crop, valid)
            mask = circle_mask(height, width, roi.cx, roi.cy, roi.radius)
            measured = measure(stack, mask, timing)
            color = channel_rgb(meta)
            write_panel(dest, crop, col0, row0, roi, vmin, vmax, timing, measured, color)
            lines = bullet_lines(meta, roi, timing, measured)
            records.append(
                {
                    "stem": path.stem + suffix,
                    "panel": str(dest),
                    "bullets": lines,
                    "diameter_um": roi.diameter * meta["um_per_px"],
                    "post_over_pre": ratio(measured["i_post"], measured["i0"]),
                    "end_over_pre": ratio(measured["i_end"], measured["i0"]),
                }
            )
            post = records[-1]["post_over_pre"]
            end = records[-1]["end_over_pre"]
            post_s = f"{post:.2f}" if post is not None else "-"
            end_s = f"{end:.2f}" if end is not None else "-"
            print(
                f"  {records[-1]['stem']}: 直径 {records[-1]['diameter_um']:.2f} μm、"
                f"一辺 {side} px、直後 {post_s}、終了 {end_s}"
            )
    return records
