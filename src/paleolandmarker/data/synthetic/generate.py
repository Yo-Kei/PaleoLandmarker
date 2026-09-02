"""Target-aware procedural fossil landmark generator (Synthetic).

The released generator uses grayscale images, 18 keypoints, masks and bounding boxes while
changing the visual curriculum:

* six morphology domains correspond to the six real transfer tasks;
* each morphology has domain-conditioned acquisition styles;
* specimens vary in scale, placement, pose, preservation and focus;
* colony/full-specimen context and distractors are rendered when appropriate;
* samples are drawn at 2x resolution and downsampled to soften vector edges.

The module is standalone by design and writes only to its configured output directory.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from paleolandmarker.utils import seed_everything


VERSION = "2.10-preview"
BASE_SIZE = 256
RENDER_SCALE = 2
REVISION_NOTES = [
    "conodont landmarks combine six informative outline points with twelve visible internal junctions",
    "gastropod is rendered from a growing three-dimensional spiral tube with an open right aperture",
    "fusulinid orientation is restricted to a small acquisition tilt so directional semantics remain stable",
    "conodont and trilobite morphology span wider outline-parameter families",
    "conodont cusp length and width have larger lower bounds",
    "conodont blade and cusp are shaded as a continuous transversely convex 3-D surface",
    "conodont internal ridges use narrow surface-relative tones instead of a flat wireframe",
    "conodont transverse structures are also encoded as low relief in the blade height field",
    "conodont transverse ridges use paired surface-relative highlight and shadow edges",
    "trilobite bilateral parameters use shared bases with only mild paired jitter",
    "trilobite cephala are vertically diversified and always fully visible while long bodies may be cropped",
    "bryozoan colony walls use a protected minimum local contrast and width",
    "gastropod camera is aperture-facing and the spiral has greater axial cone height",
    "microtus tooth size, centre and spacing receive only small bounded local drift",
    "microtus enamel triangles are raised faceted ridges separated by structural grooves",
    "microtus enamel ridges sit on a continuous low domed three-dimensional tooth base",
    "every morphology samples multiple shared acquisition-background families",
    "bryozoan target and partial neighbours share appearance, with a small single-zooid probability",
    "all six domains use eighteen landmarks tied to explicitly rendered structures",
    "fusulinid inner cardinal landmarks sit on an actual proloculus whose relative size varies independently",
    "trilobite cephala include paired glabellar furrows, an occipital ring and marginal details",
    "trilobite cephala use a three-dimensional relief field derived from local real-data morphology: a high glabella, paired raised eyes, lower sloping cheeks, a modest border rim and recessed furrows",
    "complete trilobites continue the relief through a vaulted axial thorax and lower pleural lobes",
    "conodont, gastropod and microtus samples receive sparse landmark-safe edge loss",
    "other morphologies receive occasional low-density cracks, grains or mineral seams",
    "all domains receive sparse landmark-safe acquisition artifacts including varied scale bars, plate indices and specimen labels",
    "acquisition artifacts are kept uncommon and conodont, fusulinid and trilobite targets occupy a moderately larger image fraction",
]
DOMAINS = [
    "conodont_element",
    "fusulinid_section",
    "gastropod_shell",
    "microtus_dentition",
    "bryozoan_zooid",
    "trilobite",
]

LANDMARK_NAMES = OrderedDict(
    [
        (
            "conodont_element",
            [
                "cusp_tip_outline",
                "left_basal_shoulder_outline",
                "right_basal_shoulder_outline",
                "left_cusp_blade_transition_outline",
                "right_cusp_blade_transition_outline",
                "basal_tip_outline",
                "cusp_blade_left_axis",
                "cusp_blade_median_axis",
                "cusp_blade_right_axis",
                "upper_middle_ridge_left_axis",
                "upper_middle_ridge_median_axis",
                "upper_middle_ridge_right_axis",
                "lower_middle_ridge_left_axis",
                "lower_middle_ridge_median_axis",
                "lower_middle_ridge_right_axis",
                "basal_ridge_left_axis",
                "basal_ridge_median_axis",
                "basal_ridge_right_axis",
            ],
        ),
        (
            "fusulinid_section",
            [
                "outer_left",
                "outer_right",
                "outer_dorsal",
                "outer_ventral",
                "middle_left",
                "middle_right",
                "middle_dorsal",
                "middle_ventral",
                "inner_left",
                "inner_right",
                "inner_dorsal",
                "inner_ventral",
                "intermediate_030",
                "intermediate_090",
                "intermediate_150",
                "intermediate_210",
                "intermediate_270",
                "intermediate_330",
            ],
        ),
        (
            "gastropod_shell",
            [
                "apex",
                "spire_left",
                "spire_right",
                "body_left",
                "body_right",
                "base",
                "aperture_top",
                "aperture_outer",
                "aperture_base",
                "aperture_inner",
                "suture_left",
                "suture_right",
                "whorl_axis_20",
                "whorl_axis_32",
                "whorl_axis_44",
                "whorl_axis_58",
                "whorl_axis_74",
                "whorl_axis_90",
            ],
        ),
        (
            "microtus_dentition",
            [
                "m1_anterior_outer",
                "m1_anterior_inner",
                "m1_posterior_outer",
                "m1_posterior_inner",
                "m2_anterior_outer",
                "m2_anterior_inner",
                "m2_posterior_outer",
                "m2_posterior_inner",
                "m3_anterior_outer",
                "m3_anterior_inner",
                "m3_posterior_outer",
                "m3_posterior_inner",
                "m1_upper_central_groove",
                "m1_lower_central_groove",
                "m2_upper_central_groove",
                "m2_lower_central_groove",
                "m3_upper_central_groove",
                "m3_lower_central_groove",
            ],
        ),
        (
            "bryozoan_zooid",
            [
                "orifice_top",
                "orifice_left",
                "orifice_right",
                "orifice_base",
                "chamber_top_left",
                "chamber_top_right",
                "chamber_mid_left",
                "chamber_mid_right",
                "chamber_base_left",
                "chamber_base_right",
                "avicularium_left",
                "avicularium_right",
                "orifice_upper_left",
                "orifice_upper_right",
                "orifice_lower_left",
                "orifice_lower_right",
                "median_pore_left",
                "median_pore_right",
            ],
        ),
        (
            "trilobite",
            [
                "glabella_front",
                "glabella_rear",
                "left_eye_front",
                "right_eye_front",
                "left_eye_rear",
                "right_eye_rear",
                "left_genal_tip",
                "right_genal_tip",
                "left_suture",
                "right_suture",
                "left_posterior_margin",
                "right_posterior_margin",
                "left_anterior_glabellar_furrow",
                "right_anterior_glabellar_furrow",
                "left_middle_glabellar_furrow",
                "right_middle_glabellar_furrow",
                "left_posterior_glabellar_furrow",
                "right_posterior_glabellar_furrow",
            ],
        ),
    ]
)

STYLE_WEIGHTS = {
    "conodont_element": {
        "sem": 0.30,
        "plate": 0.22,
        "matrix": 0.23,
        "macro": 0.15,
        "thin_section": 0.10,
    },
    "fusulinid_section": {
        "thin_section": 0.25,
        "plate": 0.20,
        "matrix": 0.20,
        "sem": 0.20,
        "macro": 0.15,
    },
    "gastropod_shell": {
        "macro": 0.32,
        "plate": 0.18,
        "matrix": 0.20,
        "sem": 0.15,
        "thin_section": 0.10,
        "colony": 0.05,
    },
    "microtus_dentition": {
        "macro": 0.30,
        "sem": 0.22,
        "plate": 0.18,
        "matrix": 0.20,
        "thin_section": 0.10,
    },
    "bryozoan_zooid": {
        "colony": 0.30,
        "sem": 0.20,
        "plate": 0.15,
        "matrix": 0.20,
        "macro": 0.15,
    },
    "trilobite": {
        "matrix": 0.28,
        "plate": 0.22,
        "sem": 0.20,
        "macro": 0.18,
        "thin_section": 0.12,
    },
}

ACQUISITION_ARTIFACT_PROBABILITIES = {
    "sem": 0.14,
    "plate": 0.34,
    "thin_section": 0.17,
    "macro": 0.16,
    "colony": 0.13,
    "matrix": 0.18,
}
SECOND_ACQUISITION_ARTIFACT_PROBABILITY = 0.10

LAYOUT_RANGES = {
    "conodont_element": ((0.80, 1.08), (-16, 16), (-20, 20), (-12, 12), (-0.07, 0.07)),
    "fusulinid_section": ((0.82, 1.14), (-8, 8), (-11, 11), (-12, 12), (-0.035, 0.035)),
    "gastropod_shell": ((0.82, 1.04), (-12, 12), (-15, 15), (-15, 15), (-0.05, 0.05)),
    "microtus_dentition": ((0.78, 1.02), (-12, 12), (-12, 12), (-13, 13), (-0.04, 0.04)),
    "bryozoan_zooid": ((0.86, 1.12), (-10, 10), (-12, 12), (-13, 13), (-0.05, 0.05)),
    "trilobite": ((0.72, 1.04), (-18, 18), (-19, 19), (-17, 17), (-0.08, 0.08)),
}


def _curve(points, samples=160):
    """Closed Catmull-Rom curve through control points."""
    points = np.asarray(points, dtype=np.float32)
    extended = np.vstack((points[-1], points, points[0], points[1]))
    output = []
    segments = len(points)
    per_segment = max(4, samples // segments)
    for index in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[index - 1 : index + 3]
        for t in np.linspace(0, 1, per_segment, endpoint=False):
            output.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
                )
            )
    return np.asarray(output, dtype=np.float32)


def _ellipse(cx, cy, rx, ry, samples=120, wobble=0.0, phase=0.0, harmonic=3):
    angle = np.linspace(0, 2 * math.pi, samples, endpoint=False)
    radius = 1 + wobble * np.sin(harmonic * angle + phase)
    return np.column_stack(
        (cx + rx * radius * np.cos(angle), cy + ry * radius * np.sin(angle))
    ).astype(np.float32)


def _open_arc(cx, cy, rx, ry, start, stop, samples=80):
    angle = np.linspace(start, stop, samples)
    return np.column_stack((cx + rx * np.cos(angle), cy + ry * np.sin(angle))).astype(
        np.float32
    )


def _morphology(
    fills,
    structures,
    landmarks,
    context=(),
    context_fills=(),
    context_inlays=(),
    inlays=(),
    notes=None,
):
    return {
        "fills": [np.asarray(item, np.float32) for item in fills],
        "structures": [np.asarray(item, np.float32) for item in structures],
        "context": [np.asarray(item, np.float32) for item in context],
        "context_fills": [np.asarray(item, np.float32) for item in context_fills],
        "context_inlays": [np.asarray(item, np.float32) for item in context_inlays],
        "inlays": [np.asarray(item, np.float32) for item in inlays],
        "landmarks": np.asarray(landmarks, np.float32),
        "notes": notes or {},
    }


def _surface_from_mesh(vertices_3d, faces, rng, light=None):
    """Convert a small object-space mesh into the renderer surface contract."""
    vertices_3d = np.asarray(vertices_3d, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if light is None:
        light = np.array(
            [
                rng.uniform(-0.58, 0.12),
                rng.uniform(-0.68, -0.16),
                rng.uniform(0.68, 0.94),
            ],
            dtype=np.float32,
        )
    else:
        light = np.asarray(light, dtype=np.float32).copy()
    light /= max(np.linalg.norm(light), 1e-6)
    depths, shades = [], []
    for face in faces:
        polygon = vertices_3d[face]
        normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
        normal /= max(np.linalg.norm(normal), 1e-6)
        if normal[2] < 0:
            normal *= -1
        depths.append(float(polygon[:, 2].mean()))
        shades.append(float(0.18 + 0.82 * max(np.dot(normal, light), 0.0)))
    return {
        "vertices": vertices_3d[:, :2].astype(np.float32),
        "faces": faces,
        "depths": np.asarray(depths, dtype=np.float32),
        "shades": np.asarray(shades, dtype=np.float32),
    }, light


def _horizontal_polygon_bounds(polygon, y):
    """Return the outermost intersections of a closed polygon with a scanline."""
    polygon = np.asarray(polygon, dtype=np.float32)
    intersections = []
    for first, second in zip(polygon, np.vstack((polygon[1:], polygon[:1]))):
        y1, y2 = float(first[1]), float(second[1])
        if (y1 <= y < y2) or (y2 <= y < y1):
            fraction = (y - y1) / (y2 - y1)
            intersections.append(float(first[0] + fraction * (second[0] - first[0])))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def _distance_to_polyline(point, polyline):
    """Euclidean distance from one point to the closest polyline segment."""
    point = np.asarray(point, dtype=np.float32)
    shortest = float("inf")
    for first, second in zip(polyline[:-1], polyline[1:]):
        delta = second - first
        denominator = max(float(np.dot(delta, delta)), 1e-6)
        fraction = float(np.clip(np.dot(point - first, delta) / denominator, 0, 1))
        projection = first + fraction * delta
        shortest = min(shortest, float(np.linalg.norm(point - projection)))
    return shortest


def _convex_blade_surface(outline, ridges, rng, relief, ridge_relief):
    """Build a longitudinal blade whose medial axis rises above both margins."""
    y_min = float(outline[:, 1].min()) + 1.0
    y_max = float(outline[:, 1].max()) - 1.0
    row_y = np.linspace(y_min, y_max, 68)
    lateral = np.linspace(-1.0, 1.0, 11)
    vertices = []
    valid_rows = []
    for y in row_y:
        bounds = _horizontal_polygon_bounds(outline, float(y))
        if bounds is None or bounds[1] - bounds[0] < 0.8:
            continue
        left, right = bounds
        centre = (left + right) / 2
        half_width = (right - left) / 2
        longitudinal = (y - y_min) / max(y_max - y_min, 1e-6)
        longitudinal_bulge = 0.58 + 0.42 * math.sin(math.pi * longitudinal) ** 0.65
        row = []
        for offset in lateral:
            height = (
                relief
                * longitudinal_bulge
                * max(0.0, 1.0 - abs(float(offset)) ** 1.55)
            )
            ridge_distance = min(
                (_distance_to_polyline([centre + half_width * offset, y], ridge) for ridge in ridges),
                default=float("inf"),
            )
            height += ridge_relief * math.exp(-0.5 * (ridge_distance / 2.1) ** 2)
            row.append(len(vertices))
            vertices.append([centre + half_width * offset, y, height])
        valid_rows.append(row)
    faces = []
    for first, second in zip(valid_rows[:-1], valid_rows[1:]):
        for column in range(len(lateral) - 1):
            faces.append(
                [
                    first[column],
                    first[column + 1],
                    second[column + 1],
                    second[column],
                ]
            )
    return _surface_from_mesh(vertices, faces, rng)


def _domed_plate_mesh(outline, relief):
    """Triangulate a low dome that follows an elongated specimen outline."""
    y_min = float(outline[:, 1].min()) + 1.0
    y_max = float(outline[:, 1].max()) - 1.0
    vertices, rows = [], []
    lateral = np.linspace(-1.0, 1.0, 21)
    for y in np.linspace(y_min, y_max, 41):
        bounds = _horizontal_polygon_bounds(outline, float(y))
        if bounds is None or bounds[1] - bounds[0] < 1.0:
            continue
        left, right = bounds
        centre = (left + right) / 2
        half_width = (right - left) / 2
        vertical = (y - y_min) / max(y_max - y_min, 1e-6)
        vertical_bulge = max(0.0, math.sin(math.pi * vertical)) ** 0.55
        row = []
        for offset in lateral:
            lateral_bulge = max(0.0, 1.0 - float(offset) ** 2) ** 0.60
            row.append(len(vertices))
            vertices.append(
                [centre + half_width * offset, y, relief * vertical_bulge * lateral_bulge]
            )
        rows.append(row)
    faces = []
    for first, second in zip(rows[:-1], rows[1:]):
        for column in range(len(lateral) - 1):
            faces.extend(
                [
                    [first[column], first[column + 1], second[column + 1]],
                    [first[column], second[column + 1], second[column]],
                ]
            )
    return vertices, faces, (y_min, y_max)


def _domed_plate_height(outline, point, relief, y_range):
    """Evaluate the plate height so raised structures attach to its surface."""
    y_min, y_max = y_range
    x, y = map(float, point)
    bounds = _horizontal_polygon_bounds(outline, y)
    if bounds is None:
        return 0.0
    left, right = bounds
    centre = (left + right) / 2
    half_width = max((right - left) / 2, 1e-6)
    lateral = np.clip((x - centre) / half_width, -1.0, 1.0)
    vertical = np.clip((y - y_min) / max(y_max - y_min, 1e-6), 0.0, 1.0)
    return float(
        relief
        * max(0.0, math.sin(math.pi * vertical)) ** 0.55
        * max(0.0, 1.0 - lateral**2) ** 0.60
    )


def _dentition_surface(outline, triangles, rng, base_relief, enamel_relief):
    """Combine a domed tooth base and attached rounded enamel ridges."""
    vertices, faces, y_range = _domed_plate_mesh(outline, base_relief)
    for triangle in triangles:
        triangle = np.asarray(triangle, dtype=np.float32)
        local_relief = enamel_relief * rng.uniform(0.88, 1.12)
        attachment = np.asarray(
            [
                _domed_plate_height(outline, point, base_relief, y_range)
                for point in triangle
            ],
            dtype=np.float32,
        )
        base = len(vertices)
        # The anatomical outer apex retains some height, while the interior
        # point is highest; shared endpoints remain low and read as grooves.
        vertices.extend(
            [
                [triangle[0, 0], triangle[0, 1], attachment[0] + local_relief * 0.05],
                [triangle[1, 0], triangle[1, 1], attachment[1] + local_relief * 0.42],
                [triangle[2, 0], triangle[2, 1], attachment[2] + local_relief * 0.05],
            ]
        )
        peak = triangle.mean(axis=0) * 0.45 + triangle[1] * 0.55
        peak_base = _domed_plate_height(outline, peak, base_relief, y_range)
        vertices.append([peak[0], peak[1], peak_base + local_relief])
        # An inset shoulder ring breaks the former single pyramid into a
        # gradual enamel dome.  The outer apex side stays slightly taller,
        # preserving the directional ridge rather than becoming a blob.
        shoulder_start = len(vertices)
        for index, point in enumerate(triangle):
            shoulder = point * 0.52 + peak * 0.48
            shoulder_base = _domed_plate_height(
                outline, shoulder, base_relief, y_range
            )
            shoulder_height = local_relief * (0.62 if index != 1 else 0.76)
            vertices.append(
                [shoulder[0], shoulder[1], shoulder_base + shoulder_height]
            )
        peak_index = base + 3
        shoulder = [shoulder_start, shoulder_start + 1, shoulder_start + 2]
        faces.extend(
            [
                [base, base + 1, shoulder[1]],
                [base, shoulder[1], shoulder[0]],
                [base + 1, base + 2, shoulder[2]],
                [base + 1, shoulder[2], shoulder[1]],
                [base + 2, base, shoulder[0]],
                [base + 2, shoulder[0], shoulder[2]],
                [shoulder[0], shoulder[1], peak_index],
                [shoulder[1], shoulder[2], peak_index],
                [shoulder[2], shoulder[0], peak_index],
            ]
        )
    return _surface_from_mesh(vertices, faces, rng)


def _trilobite_cephalon_surface(
    outline,
    glabella_center,
    glabella_radii,
    eye_centers,
    eye_radii,
    furrows,
    sutures,
    rng,
    cheek_relief,
    glabella_relief,
    eye_relief,
    furrow_depth,
    border_relief,
):
    """Build a cephalon relief field following the local real-image anatomy.

    The selected Trilomorph photographs consistently show a dominant axial
    glabellar dome, smaller raised eye lobes and lower fixigenae that fall
    towards the margin.  Glabellar furrows and facial sutures are therefore
    encoded as shallow depressions in the same continuous surface instead of
    being pasted-on dark curves.
    """
    outline = np.asarray(outline, dtype=np.float32)
    glabella_center = np.asarray(glabella_center, dtype=np.float32)
    glabella_radii = np.asarray(glabella_radii, dtype=np.float32)
    eye_centers = [np.asarray(point, dtype=np.float32) for point in eye_centers]
    eye_radii = [np.asarray(radii, dtype=np.float32) for radii in eye_radii]
    y_min = float(outline[:, 1].min()) + 1.0
    y_max = float(outline[:, 1].max()) - 1.0
    lateral = np.linspace(-1.0, 1.0, 31)
    vertices, rows = [], []
    relief_phase_x, relief_phase_y = rng.uniform(0, 2 * math.pi, 2)

    for y in np.linspace(y_min, y_max, 55):
        bounds = _horizontal_polygon_bounds(outline, float(y))
        if bounds is None or bounds[1] - bounds[0] < 1.2:
            continue
        left, right = bounds
        centre = (left + right) / 2
        half_width = max((right - left) / 2, 1e-6)
        vertical = (y - y_min) / max(y_max - y_min, 1e-6)
        vertical_cap = max(0.0, math.sin(math.pi * vertical)) ** 0.48
        row = []
        for offset in lateral:
            x = centre + half_width * float(offset)
            point = np.array([x, y], dtype=np.float32)

            # Broad low cheek dome.  It is continuous across the cephalon but
            # slopes strongly towards the lateral and anterior/posterior rims.
            lateral_cap = max(0.0, 1.0 - abs(float(offset)) ** 1.7) ** 0.62
            height = cheek_relief * vertical_cap * lateral_cap

            glab_dx = (x - glabella_center[0]) / max(glabella_radii[0], 1e-6)
            glab_dy = (y - glabella_center[1]) / max(glabella_radii[1], 1e-6)
            height += glabella_relief * math.exp(
                -0.5 * (glab_dx**2 / 0.78**2 + glab_dy**2 / 0.88**2)
            )

            for eye_center, radii in zip(eye_centers, eye_radii):
                eye_dx = (x - eye_center[0]) / max(radii[0], 1e-6)
                eye_dy = (y - eye_center[1]) / max(radii[1], 1e-6)
                height += eye_relief * math.exp(
                    -0.5 * (eye_dx**2 / 0.82**2 + eye_dy**2 / 0.86**2)
                )

            # A modest doublure/border roll is common, but it remains below
            # the eye and axial lobes so it cannot read as a flat embossed rim.
            rim_distance = (abs(float(offset)) - 0.91) / 0.065
            height += border_relief * vertical_cap * math.exp(-0.5 * rim_distance**2)

            for furrow in furrows:
                distance = _distance_to_polyline(point, furrow)
                height -= furrow_depth * math.exp(-0.5 * (distance / 2.2) ** 2)
            for suture in sutures:
                distance = _distance_to_polyline(point, suture)
                height -= furrow_depth * 0.38 * math.exp(
                    -0.5 * (distance / 2.0) ** 2
                )

            # Low-amplitude coherent microrelief evokes casts and granular
            # cuticle without replacing the anatomical height hierarchy.
            microrelief = 0.34 * math.sin(x * 0.17 + relief_phase_x) * math.cos(
                y * 0.14 + relief_phase_y
            )
            height = max(0.0, height + microrelief * vertical_cap * lateral_cap)
            row.append(len(vertices))
            vertices.append([x, y, height])
        rows.append(row)

    faces = []
    for first, second in zip(rows[:-1], rows[1:]):
        for column in range(len(lateral) - 1):
            faces.append(
                [
                    first[column],
                    first[column + 1],
                    second[column + 1],
                    second[column],
                ]
            )
    return _surface_from_mesh(vertices, faces, rng)


def _trilobite_thorax_surface(
    outline, segment_y, rng, pleural_relief, axial_relief, light
):
    """Give complete individuals a vaulted axis and sloping pleural lobes."""
    outline = np.asarray(outline, dtype=np.float32)
    y_min = float(outline[:, 1].min()) + 1.0
    y_max = float(outline[:, 1].max()) - 1.0
    lateral = np.linspace(-1.0, 1.0, 27)
    vertices, rows = [], []
    for y in np.linspace(y_min, y_max, 64):
        bounds = _horizontal_polygon_bounds(outline, float(y))
        if bounds is None or bounds[1] - bounds[0] < 1.0:
            continue
        left, right = bounds
        centre = (left + right) / 2
        half_width = max((right - left) / 2, 1e-6)
        vertical = np.clip((y - y_min) / max(y_max - y_min, 1e-6), 0, 1)
        taper = max(0.0, math.sin(math.pi * vertical)) ** 0.42
        row = []
        for offset in lateral:
            x = centre + half_width * float(offset)
            pleural_cap = max(0.0, 1.0 - abs(float(offset)) ** 1.55) ** 0.72
            axial_lobe = math.exp(-0.5 * (float(offset) / 0.24) ** 2)
            height = taper * (pleural_relief * pleural_cap + axial_relief * axial_lobe)
            if len(segment_y):
                distance = float(np.min(np.abs(np.asarray(segment_y) - y)))
                height += (
                    0.18
                    * pleural_relief
                    * math.exp(-0.5 * (distance / 2.0) ** 2)
                    * pleural_cap
                )
            row.append(len(vertices))
            vertices.append([x, y, max(0.0, height)])
        rows.append(row)
    faces = []
    for first, second in zip(rows[:-1], rows[1:]):
        for column in range(len(lateral) - 1):
            faces.append(
                [
                    first[column],
                    first[column + 1],
                    second[column + 1],
                    second[column],
                ]
            )
    return _surface_from_mesh(vertices, faces, rng, light=light)[0]


def _merge_surfaces(first, second):
    offset = len(first["vertices"])
    return {
        "vertices": np.vstack((first["vertices"], second["vertices"])).astype(
            np.float32
        ),
        "faces": np.vstack((first["faces"], second["faces"] + offset)).astype(
            np.int32
        ),
        "depths": np.concatenate((first["depths"], second["depths"])).astype(
            np.float32
        ),
        "shades": np.concatenate((first["shades"], second["shades"])).astype(
            np.float32
        ),
    }


def conodont_element(rng):
    cx = 128 + rng.uniform(-4, 4)
    top = rng.uniform(7, 25)
    cusp_length = rng.uniform(92, 142)
    cusp_bottom = min(top + cusp_length, 165)
    base_bottom = rng.uniform(max(cusp_bottom + 70, 218), 246)
    lean = rng.uniform(-18, 18)
    body_width = rng.uniform(34, 67)
    left_base = body_width * rng.uniform(0.78, 1.14)
    right_base = body_width * rng.uniform(0.78, 1.18)
    cusp_half = rng.uniform(14, min(28, max(16, body_width * 0.55)))
    basal_rounding = rng.uniform(13, min(34, body_width * 0.58))
    controls = [
        [cx + lean, top],
        [cx + cusp_half * 0.44, top + (cusp_bottom - top) * 0.36],
        [cx + cusp_half, cusp_bottom - 14],
        [cx + right_base, base_bottom - 54],
        [cx + basal_rounding, base_bottom - 4],
        [cx + rng.uniform(-5, 7), base_bottom],
        [cx - basal_rounding * rng.uniform(0.90, 1.12), base_bottom - rng.uniform(3, 11)],
        [cx - left_base, base_bottom - 59],
        [cx - cusp_half * rng.uniform(0.90, 1.10), cusp_bottom - 17],
        [cx - cusp_half * 0.42, top + (cusp_bottom - top) * 0.39],
    ]
    outline = _curve(controls, 180)
    structures = []
    ridge_triplets = []
    denticle_count = int(rng.integers(10, 17))
    for index, y in enumerate(np.linspace(cusp_bottom + 4, base_bottom - 39, denticle_count)):
        fraction = index / max(denticle_count - 1, 1)
        half = (1 - fraction) * cusp_half + fraction * min(left_base, right_base) * 0.82
        offset = rng.uniform(-2.5, 2.5)
        triplet = np.array(
            [
                [cx - half, y + rng.uniform(-1.5, 1.5)],
                [cx + offset, y - rng.uniform(5, 9)],
                [cx + half, y + rng.uniform(-1.5, 1.5)],
            ]
        )
        structures.append(triplet)
        ridge_triplets.append(triplet)

    # Three longitudinal axes explicitly pass through every transverse ridge.
    # Landmarks are sampled only at these visible junctions, never on empty
    # silhouette locations.  One selected level straddles the cusp/blade
    # transition and the others cover upper, middle and basal blade structure.
    ridge_triplets = np.asarray(ridge_triplets, dtype=np.float32)
    for column in range(3):
        structures.append(ridge_triplets[:, column, :])
    selected = np.array(
        [
            int(round((denticle_count - 1) * 0.18)),
            int(round((denticle_count - 1) * 0.42)),
            int(round((denticle_count - 1) * 0.70)),
            denticle_count - 1,
        ],
        dtype=int,
    )
    contour_targets = np.array(
        [
            [cx + lean, top],
            [cx - left_base, base_bottom - 59],
            [cx + right_base, base_bottom - 54],
            [cx - cusp_half * 0.96, cusp_bottom - 17],
            [cx + cusp_half, cusp_bottom - 14],
            [cx, base_bottom],
        ],
        dtype=np.float32,
    )
    contour_landmarks = []
    for target in contour_targets:
        contour_landmarks.append(outline[np.argmin(np.linalg.norm(outline - target, axis=1))])
    landmarks = np.vstack(
        [np.asarray(contour_landmarks, dtype=np.float32), ridge_triplets[selected].reshape(12, 2)]
    )
    surface_relief = rng.uniform(17, 31)
    ridge_surface_relief = rng.uniform(1.8, 3.4)
    surface, light = _convex_blade_surface(
        outline, ridge_triplets, rng, surface_relief, ridge_surface_relief
    )
    morphology = _morphology(
        [outline],
        structures,
        landmarks,
        notes={
            "denticle_count": denticle_count,
            "body_width": float(body_width),
            "cusp_length": float(cusp_bottom - top),
            "cusp_half_width": float(cusp_half),
            "geometry": "three_dimensional_transversely_convex_blade",
            "surface_relief": float(surface_relief),
            "ridge_surface_relief": float(ridge_surface_relief),
            "surface_light_direction": light.tolist(),
            "landmark_rule": "six_outline_points_plus_four_visible_ridges_x_three_longitudinal_axes",
            "landmark_roles": ["outline"] * 6 + ["internal_junction"] * 12,
        },
    )
    morphology["surface"] = surface
    morphology["structure_kinds"] = ["transverse_ridge"] * denticle_count + [
        "longitudinal_axis"
    ] * 3
    return morphology


def fusulinid_section(rng):
    cx, cy = 128 + rng.uniform(-4, 4), 128 + rng.uniform(-4, 4)
    rx = rng.uniform(78, 112)
    ry = rx * rng.uniform(0.28, 0.72)
    if rng.random() < 0.18:
        ry = rx * rng.uniform(0.78, 1.02)
    outline = _ellipse(
        cx,
        cy,
        rx,
        ry,
        180,
        wobble=rng.uniform(0.01, 0.06),
        phase=rng.uniform(0, 6),
        harmonic=int(rng.integers(3, 7)),
    )
    structures = []
    ring_count = int(rng.integers(6, 10))
    proloculus_fraction = rng.uniform(0.09, 0.22)
    second_ring_fraction = min(
        proloculus_fraction + rng.uniform(0.07, 0.13), 0.34
    )
    ring_fractions = np.concatenate(
        (
            [proloculus_fraction],
            np.linspace(second_ring_fraction, 0.90, ring_count - 1),
        )
    )
    for ring_index, fraction in enumerate(ring_fractions):
        # The first ring is the proloculus.  Keep its realised outline so the
        # four inner landmarks can be attached to the actual rendered chamber
        # rather than to an unrelated larger nominal ellipse.
        ring_cx = (
            cx + rng.uniform(-1.0, 1.0)
            if ring_index == 0
            else cx + rng.uniform(-2, 2)
        )
        ring_cy = (
            cy + rng.uniform(-0.8, 0.8)
            if ring_index == 0
            else cy + rng.uniform(-1.5, 1.5)
        )
        ring = _ellipse(
            ring_cx,
            ring_cy,
            rx * fraction,
            ry * fraction,
            140,
            wobble=rng.uniform(0.01, 0.04 if ring_index == 0 else 0.05),
            phase=rng.uniform(0, 6),
            harmonic=int(rng.integers(3, 8)),
        )
        structures.append(ring)
    proloculus = structures[0]
    chamber_count = int(rng.integers(26, 44))
    for angle in np.linspace(0, 2 * math.pi, chamber_count, endpoint=False):
        jitter = rng.uniform(-0.08, 0.08)
        direction = angle + jitter
        inner = rng.uniform(0.10, 0.22)
        outer = rng.uniform(0.82, 0.98)
        bend = rng.uniform(-0.10, 0.10)
        middle = rng.uniform(0.42, 0.72)
        structures.append(
            np.array(
                [
                    [cx + rx * inner * math.cos(direction), cy + ry * inner * math.sin(direction)],
                    [cx + rx * middle * math.cos(direction + bend), cy + ry * middle * math.sin(direction + bend)],
                    [cx + rx * outer * math.cos(direction - bend * 0.4), cy + ry * outer * math.sin(direction - bend * 0.4)],
                ]
            )
        )
    landmark_fraction = 0.46
    protected_structure_indices = [0]
    landmark_ring = _ellipse(cx, cy, rx * landmark_fraction, ry * landmark_fraction, 140)
    protected_structure_indices.append(len(structures))
    structures.append(landmark_ring)
    diagonal_landmarks = []
    for angle in np.deg2rad([30, 90, 150, 210, 270, 330]):
        point = np.array(
            [
                cx + rx * landmark_fraction * math.cos(angle),
                cy + ry * landmark_fraction * math.sin(angle),
            ],
            dtype=np.float32,
        )
        diagonal_landmarks.append(point)
        protected_structure_indices.append(len(structures))
        structures.append(
            np.array(
                [
                    [cx + rx * 0.38 * math.cos(angle), cy + ry * 0.38 * math.sin(angle)],
                    point,
                    [cx + rx * 0.54 * math.cos(angle), cy + ry * 0.54 * math.sin(angle)],
                ],
                dtype=np.float32,
            )
        )
    landmarks = []
    for fraction in (1.0, 0.62):
        landmarks.extend(
            [
                [cx - rx * fraction, cy],
                [cx + rx * fraction, cy],
                [cx, cy - ry * fraction],
                [cx, cy + ry * fraction],
            ]
        )
    landmarks.extend(
        [
            proloculus[np.argmin(proloculus[:, 0])],
            proloculus[np.argmax(proloculus[:, 0])],
            proloculus[np.argmin(proloculus[:, 1])],
            proloculus[np.argmax(proloculus[:, 1])],
        ]
    )
    landmarks.extend(diagonal_landmarks)
    morphology = _morphology(
        [outline],
        structures,
        landmarks,
        notes={
            "chamber_count": chamber_count,
            "aspect_ratio": ry / rx,
            "proloculus_structure_index": 0,
            "inner_landmark_rule": "actual_proloculus_cardinal_extrema",
            "proloculus_fraction": float(proloculus_fraction),
            "proloculus_fraction_sampling_range": [0.09, 0.22],
        },
    )
    morphology["protected_structure_indices"] = protected_structure_indices
    return morphology


def _gastropod_shell_2d_legacy(rng):
    # The whorl opens from a centre on the left and terminates at a right-side
    # aperture.  One parameter set controls the logarithmic coil, outline and
    # aperture, preventing impossible combinations of shell and spiral.
    whorl_cx = rng.uniform(101, 113)
    whorl_cy = rng.uniform(122, 134)
    outer_radius = rng.uniform(51, 66)
    x_aspect = rng.uniform(1.14, 1.31)
    y_aspect = rng.uniform(0.88, 1.04)
    turns = rng.uniform(2.05, 2.45)
    theta = np.linspace(-turns * 2 * math.pi, 0.0, 280)
    radius = np.exp(
        np.linspace(math.log(rng.uniform(2.0, 3.5)), math.log(outer_radius), len(theta))
    )
    spiral = np.column_stack(
        (
            whorl_cx + radius * x_aspect * np.cos(theta),
            whorl_cy + radius * y_aspect * np.sin(theta),
        )
    )

    left = whorl_cx - outer_radius * x_aspect * rng.uniform(0.70, 0.82)
    top = whorl_cy - outer_radius * y_aspect * rng.uniform(0.72, 0.83)
    bottom = whorl_cy + outer_radius * y_aspect * rng.uniform(0.70, 0.82)
    terminal_x = float(spiral[-1, 0])
    aperture_rx = outer_radius * rng.uniform(0.31, 0.40)
    aperture_ry = outer_radius * y_aspect * rng.uniform(0.76, 0.90)
    aperture_x = terminal_x + aperture_rx * rng.uniform(0.08, 0.20)
    aperture_y = whorl_cy + outer_radius * y_aspect * rng.uniform(0.08, 0.18)
    right = aperture_x + aperture_rx
    apex = np.array([whorl_cx - outer_radius * 0.05, top], dtype=np.float32)

    outline = _curve(
        [
            [left, whorl_cy + outer_radius * 0.04],
            [whorl_cx - outer_radius * x_aspect * 0.62, top + outer_radius * 0.22],
            [whorl_cx - outer_radius * x_aspect * 0.30, top + outer_radius * 0.06],
            apex,
            [whorl_cx + outer_radius * x_aspect * 0.40, top - outer_radius * 0.04],
            [aperture_x - aperture_rx * 0.20, aperture_y - aperture_ry],
            [right, aperture_y],
            [aperture_x - aperture_rx * 0.18, aperture_y + aperture_ry],
            [whorl_cx + outer_radius * x_aspect * 0.34, bottom + outer_radius * 0.04],
            [whorl_cx - outer_radius * x_aspect * 0.24, bottom],
            [whorl_cx - outer_radius * x_aspect * 0.61, bottom - outer_radius * 0.18],
        ],
        240,
    )

    aperture = _ellipse(
        aperture_x,
        aperture_y,
        aperture_rx,
        aperture_ry,
        110,
        wobble=rng.uniform(0.01, 0.05),
        phase=rng.uniform(0, 6),
    )

    structures = [spiral, aperture]
    growth_band_count = int(rng.integers(4, 8))
    for fraction in np.linspace(0.34, 0.88, growth_band_count):
        band_x = whorl_cx + (aperture_x - whorl_cx) * fraction
        band_half = outer_radius * y_aspect * (0.38 + 0.38 * fraction)
        bow = outer_radius * rng.uniform(0.035, 0.075)
        structures.append(
            np.array(
                [
                    [band_x - bow, whorl_cy - band_half],
                    [band_x + bow, whorl_cy - band_half * 0.28],
                    [band_x + bow, whorl_cy + band_half * 0.35],
                    [band_x - bow, whorl_cy + band_half],
                ]
            )
        )

    # Soft body is a low-contrast background object and may be only partly
    # visible, matching the photographs in the real dataset.
    tail_x = left + outer_radius * 0.08
    head_x = right + outer_radius * 0.16
    body = _curve(
        [
            [tail_x, whorl_cy + outer_radius * y_aspect * 0.64],
            [whorl_cx, whorl_cy + outer_radius * y_aspect * 0.88],
            [aperture_x, whorl_cy + outer_radius * y_aspect * 0.88],
            [head_x, whorl_cy + outer_radius * y_aspect * 0.71],
            [aperture_x, whorl_cy + outer_radius * y_aspect * 0.58],
            [whorl_cx, whorl_cy + outer_radius * y_aspect * 0.54],
        ],
        140,
    )
    head = _ellipse(
        head_x,
        whorl_cy + outer_radius * y_aspect * 0.68,
        outer_radius * 0.18,
        outer_radius * 0.12,
        60,
    )
    tentacles = []
    for vertical in (0.39, 0.46):
        tentacles.append(
            np.array(
                [
                    [head_x, whorl_cy + outer_radius * y_aspect * 0.62],
                    [
                        head_x + outer_radius * rng.uniform(0.17, 0.25),
                        whorl_cy + outer_radius * y_aspect * vertical,
                    ],
                ]
            )
        )

    landmarks = [
        apex,
        [whorl_cx - outer_radius * x_aspect * 0.52, top + outer_radius * 0.14],
        [whorl_cx + outer_radius * x_aspect * 0.40, top - outer_radius * 0.04],
        [left, whorl_cy + outer_radius * 0.04],
        [right, aperture_y],
        [whorl_cx - outer_radius * x_aspect * 0.22, bottom],
        [aperture_x, aperture_y - aperture_ry],
        [aperture_x + aperture_rx, aperture_y],
        [aperture_x, aperture_y + aperture_ry],
        [aperture_x - aperture_rx, aperture_y],
        spiral[len(spiral) // 2],
        spiral[-1],
    ]
    return _morphology(
        [outline],
        structures,
        landmarks,
        context=tentacles,
        context_fills=[body, head],
        inlays=[aperture],
        notes={
            "coil_direction": "left",
            "aperture_side": "right",
            "whorl_turns": float(turns),
            "outer_radius": float(outer_radius),
            "whorl_aspect": [float(x_aspect), float(y_aspect)],
            "whorl_center": [float(whorl_cx), float(whorl_cy)],
            "aperture_center": [float(aperture_x), float(aperture_y)],
            "growth_band_count": growth_band_count,
        },
    )


def _convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=np.float32))))
    if len(points) <= 2:
        return np.asarray(points, dtype=np.float32)

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def gastropod_shell(rng):
    """Render a growing 3-D spiral tube, viewed obliquely from above."""
    turns = rng.uniform(2.15, 2.75)
    outer_radius = rng.uniform(48, 64)
    inner_radius = rng.uniform(2.5, 5.0)
    aperture_radius = outer_radius * rng.uniform(0.35, 0.48)
    spire_height = rng.uniform(48, 82)
    radial_power = rng.uniform(1.25, 1.75)
    tube_power = rng.uniform(1.30, 1.75)
    longitudinal_scale = rng.uniform(1.02, 1.22)
    n_u, n_phi = 72, 30
    u = np.linspace(0, 1, n_u)
    angle = -turns * 2 * math.pi * (1 - u)
    rho = inner_radius + (outer_radius - inner_radius) * u**radial_power
    tube_radius = rng.uniform(2.2, 4.0) + (
        aperture_radius - rng.uniform(2.2, 4.0)
    ) * u**tube_power
    centers = np.column_stack(
        (
            rho * np.cos(angle) * longitudinal_scale,
            rho * np.sin(angle),
            spire_height * (1 - u) ** rng.uniform(0.78, 0.98),
        )
    )
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    n1 = np.column_stack((-tangents[:, 1], tangents[:, 0], np.zeros(n_u)))
    n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-6)
    n2 = np.cross(tangents, n1)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-6)
    phi = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    vertices = np.empty((n_u, n_phi, 3), dtype=np.float32)
    vertex_normals = np.empty_like(vertices)
    for index, phase in enumerate(phi):
        radial_normal = math.cos(phase) * n1 + math.sin(phase) * n2
        vertices[:, index] = centers + tube_radius[:, None] * radial_normal
        vertex_normals[:, index] = radial_normal

    # View from near the aperture normal.  Small radial and vertical offsets
    # retain some body-whorl context while the opening remains the dominant
    # viewing direction.
    aperture_normal = tangents[-1].copy()
    radial = centers[-1].copy()
    radial[2] = 0
    radial /= max(np.linalg.norm(radial), 1e-6)
    forward = (
        aperture_normal
        + radial * rng.uniform(-0.14, 0.14)
        + np.array([0, 0, rng.uniform(0.06, 0.18)], dtype=np.float32)
    )
    forward /= max(np.linalg.norm(forward), 1e-6)
    world_up = np.array([0, 0, 1], dtype=np.float32)
    camera_right = np.cross(world_up, forward)
    camera_right /= max(np.linalg.norm(camera_right), 1e-6)
    camera_up = np.cross(forward, camera_right)
    camera_up /= max(np.linalg.norm(camera_up), 1e-6)
    rotation = np.vstack((camera_right, -camera_up, forward)).astype(np.float32)
    camera_vertices = vertices @ rotation.T
    camera_normals = vertex_normals @ rotation.T
    center_camera = centers @ rotation.T
    if camera_vertices[-1, :, 0].mean() < camera_vertices[..., 0].mean():
        camera_vertices[..., 0] *= -1
        camera_normals[..., 0] *= -1
        center_camera[..., 0] *= -1
    projected = camera_vertices[..., :2].copy()
    minimum = projected.reshape(-1, 2).min(axis=0)
    maximum = projected.reshape(-1, 2).max(axis=0)
    extent = maximum - minimum
    scale = min(rng.uniform(176, 204) / max(extent[0], 1), rng.uniform(166, 204) / max(extent[1], 1))
    projected = (projected - (minimum + maximum) / 2) * scale
    projected[..., 0] += rng.uniform(118, 128)
    projected[..., 1] += rng.uniform(124, 137)
    projected_flat = projected.reshape(-1, 2).astype(np.float32)

    faces, depths, shades = [], [], []
    light = np.array([-0.35, -0.45, 0.82], dtype=np.float32)
    light /= np.linalg.norm(light)
    for row in range(n_u - 1):
        for column in range(n_phi):
            next_column = (column + 1) % n_phi
            face = [
                row * n_phi + column,
                (row + 1) * n_phi + column,
                (row + 1) * n_phi + next_column,
                row * n_phi + next_column,
            ]
            normal = camera_normals[[row, row + 1], [column, next_column]].mean(axis=0)
            normal /= max(np.linalg.norm(normal), 1e-6)
            faces.append(face)
            depths.append(float(camera_vertices[[row, row + 1], [column, next_column], 2].mean()))
            shades.append(float(0.20 + 0.80 * max(np.dot(normal, light), 0.0)))

    hull = _convex_hull(projected_flat)
    center_projected = (center_camera[:, :2] - (minimum + maximum) / 2) * scale
    center_projected[:, 0] += projected[..., 0].mean() - center_projected[:, 0].mean()
    center_projected[:, 1] += projected[..., 1].mean() - center_projected[:, 1].mean()
    aperture_ring = projected[-1]
    structures = [center_projected.astype(np.float32), aperture_ring.astype(np.float32)]
    for fraction in np.linspace(0.55, 0.90, int(rng.integers(3, 6))):
        structures.append(projected[int(round(fraction * (n_u - 1)))].astype(np.float32))

    def hull_extreme(axis, maximum_value):
        values = hull[:, axis]
        return hull[np.argmax(values) if maximum_value else np.argmin(values)]

    aperture_left = aperture_ring[np.argmin(aperture_ring[:, 0])]
    aperture_right = aperture_ring[np.argmax(aperture_ring[:, 0])]
    aperture_top = aperture_ring[np.argmin(aperture_ring[:, 1])]
    aperture_bottom = aperture_ring[np.argmax(aperture_ring[:, 1])]
    landmarks = np.vstack(
        [
            hull_extreme(0, False),
            hull_extreme(1, False),
            hull_extreme(1, True),
            hull_extreme(0, True),
            aperture_top,
            aperture_right,
            aperture_bottom,
            aperture_left,
            center_projected[int(n_u * 0.34)],
            center_projected[int(n_u * 0.52)],
            center_projected[int(n_u * 0.70)],
            center_projected[int(n_u * 0.86)],
            center_projected[int(n_u * 0.20)],
            center_projected[int(n_u * 0.32)],
            center_projected[int(n_u * 0.44)],
            center_projected[int(n_u * 0.58)],
            center_projected[int(n_u * 0.74)],
            center_projected[int(n_u * 0.90)],
        ]
    ).astype(np.float32)

    right = float(hull[:, 0].max())
    bottom = float(hull[:, 1].max())
    body = _curve(
        [
            [float(hull[:, 0].min()) + 12, bottom - 12],
            [projected[..., 0].mean(), bottom + 5],
            [right + 25, bottom + 2],
            [right + 36, bottom - 10],
            [right + 15, bottom - 18],
            [projected[..., 0].mean(), bottom - 15],
        ],
        120,
    )
    morphology = _morphology(
        [hull],
        structures,
        landmarks,
        context_fills=[body],
        inlays=[aperture_ring],
        notes={
            "geometry": "three_dimensional_growing_spiral_tube",
            "coil_direction": "left",
            "aperture_side": "right",
            "whorl_turns": float(turns),
            "outer_radius": float(outer_radius),
            "aperture_radius": float(aperture_radius),
            "spire_height": float(spire_height),
            "view_direction": "aperture_facing_oblique",
            "aperture_view_deviation_degrees": float(
                math.degrees(
                    math.acos(float(np.clip(np.dot(forward, aperture_normal), -1, 1)))
                )
            ),
        },
    )
    morphology["surface"] = {
        "vertices": projected_flat,
        "faces": np.asarray(faces, dtype=np.int32),
        "depths": np.asarray(depths, dtype=np.float32),
        "shades": np.asarray(shades, dtype=np.float32),
    }
    morphology["protected_structure_indices"] = [0]
    return morphology


def microtus_dentition(rng):
    cx, cy = 128 + rng.uniform(-6, 6), 130 + rng.uniform(-5, 5)
    total_width = rng.uniform(188, 226)
    half_height = rng.uniform(51, 68)
    left = cx - total_width / 2
    right = cx + total_width / 2
    controls = [
        [left, cy - half_height * 0.16],
        [left + total_width * 0.10, cy - half_height * 0.72],
        [cx - total_width * 0.20, cy - half_height],
        [cx + total_width * 0.38, cy - half_height * 0.86],
        [right, cy - half_height * 0.34],
        [right - total_width * 0.03, cy + half_height * 0.52],
        [cx + total_width * 0.24, cy + half_height * 0.80],
        [cx - total_width * 0.25, cy + half_height * 0.72],
        [left + total_width * 0.05, cy + half_height * 0.42],
    ]
    outline = _curve(controls, 200)
    structures = []
    enamel_triangles = []
    landmarks = []
    groove_landmarks = []
    nominal_centers = [left + total_width * fraction for fraction in (0.22, 0.50, 0.78)]
    nominal_widths = [total_width * value for value in (0.23, 0.24, 0.20)]
    nominal_outer_depth = half_height * rng.uniform(0.52, 0.58)
    nominal_inner_depth = half_height * rng.uniform(0.54, 0.60)
    drift_records = []
    for nominal_x, nominal_width in zip(nominal_centers, nominal_widths):
        center_x = nominal_x + rng.uniform(-total_width * 0.009, total_width * 0.009)
        center_y = cy + rng.uniform(-half_height * 0.025, half_height * 0.025)
        tooth_width = nominal_width * rng.uniform(0.95, 1.05)
        gap_bias = rng.uniform(-tooth_width * 0.012, tooth_width * 0.012)
        outer = center_y - nominal_outer_depth * rng.uniform(0.96, 1.04)
        inner = center_y + nominal_inner_depth * rng.uniform(0.96, 1.04)
        left_peak = center_x - tooth_width * rng.uniform(0.255, 0.285) + gap_bias
        right_peak = center_x + tooth_width * rng.uniform(0.255, 0.285) + gap_bias
        upper = np.array(
            [
                [center_x - tooth_width * rng.uniform(0.505, 0.535), center_y + half_height * rng.uniform(0.02, 0.05)],
                [left_peak, outer],
                [center_x + gap_bias, center_y + half_height * rng.uniform(-0.01, 0.06)],
                [right_peak, outer + rng.uniform(-half_height * 0.018, half_height * 0.018)],
                [center_x + tooth_width * rng.uniform(0.505, 0.535), center_y + half_height * rng.uniform(0.02, 0.05)],
            ]
        )
        lower_left_peak = left_peak + rng.uniform(-1.0, 1.0)
        lower_right_peak = right_peak + rng.uniform(-1.0, 1.0)
        lower_right_y = inner + rng.uniform(-half_height * 0.018, half_height * 0.018)
        lower = np.array(
            [
                [center_x - tooth_width * rng.uniform(0.505, 0.535), center_y + half_height * rng.uniform(0.00, 0.04)],
                [lower_left_peak, inner],
                [center_x + gap_bias, center_y + half_height * rng.uniform(-0.05, 0.02)],
                [lower_right_peak, lower_right_y],
                [center_x + tooth_width * rng.uniform(0.505, 0.535), center_y + half_height * rng.uniform(0.00, 0.04)],
            ]
        )
        structures.extend([upper, lower])
        enamel_triangles.extend(
            [upper[[0, 1, 2]], upper[[2, 3, 4]], lower[[0, 1, 2]], lower[[2, 3, 4]]]
        )
        landmarks.extend(
            [
                [left_peak, outer],
                [lower_left_peak, inner],
                [right_peak, outer],
                [lower_right_peak, lower_right_y],
            ]
        )
        groove_landmarks.extend([upper[2], lower[2]])
        drift_records.append(
            {
                "center_dx": float(center_x - nominal_x),
                "center_dy": float(center_y - cy),
                "width_scale": float(tooth_width / nominal_width),
            }
        )
    context = []
    for _ in range(int(rng.integers(7, 16))):
        x, y = rng.uniform(0, BASE_SIZE, 2)
        radius = rng.uniform(2, 6)
        context.append(_ellipse(x, y, radius, radius * rng.uniform(0.6, 1.5), 20))
    base_relief = rng.uniform(8, 15)
    enamel_relief = rng.uniform(9, 17)
    surface, light = _dentition_surface(
        outline, enamel_triangles, rng, base_relief, enamel_relief
    )
    landmarks.extend(groove_landmarks)
    morphology = _morphology(
        [outline],
        structures,
        landmarks,
        context=context,
        notes={
            "tooth_count": 3,
            "tooth_drift": drift_records,
            "geometry": "three_dimensional_domed_base_with_enamel_ridges",
            "base_surface_relief": float(base_relief),
            "enamel_surface_relief": float(enamel_relief),
            "surface_light_direction": light.tolist(),
        },
    )
    morphology["surface"] = surface
    morphology["protected_structure_indices"] = list(range(len(structures)))
    return morphology


def _zooid_outline(cx, cy, width, height, irregularity, rng):
    controls = [
        [cx, cy - height / 2],
        [cx - width * 0.43, cy - height * 0.38],
        [cx - width / 2, cy],
        [cx - width * 0.38, cy + height * 0.43],
        [cx, cy + height / 2],
        [cx + width * 0.38, cy + height * 0.43],
        [cx + width / 2, cy],
        [cx + width * 0.43, cy - height * 0.38],
    ]
    controls = np.asarray(controls) + rng.normal(0, irregularity, (8, 2))
    return _curve(controls, 120)


def bryozoan_zooid(rng, forced_single=None):
    # A crop is centred on one labelled zooid.  Adjacent zooids belong to the
    # same colony and therefore share its tone and texture, but their centres
    # fall mostly outside the crop so only partial walls/features remain.
    cx, cy = 128 + rng.uniform(-7, 7), 130 + rng.uniform(-7, 7)
    width, height = rng.uniform(103, 128), rng.uniform(166, 204)
    outline = _zooid_outline(cx, cy, width, height, 2.5, rng)
    orifice_cx = cx + rng.uniform(-3, 3)
    orifice_cy = cy - height * 0.24
    orifice_rx, orifice_ry = width * 0.28, height * 0.15
    orifice = _ellipse(
        orifice_cx,
        orifice_cy,
        orifice_rx,
        orifice_ry,
        90,
        wobble=rng.uniform(0.01, 0.06),
        phase=rng.uniform(0, 6),
        harmonic=3,
    )
    structures = [orifice]
    median_pore_left = np.array([cx - width * 0.18, cy + height * 0.16])
    median_pore_right = np.array([cx + width * 0.18, cy + height * 0.16])
    structures.extend(
        [
            _ellipse(*median_pore_left, width * 0.035, height * 0.022, 24),
            _ellipse(*median_pore_right, width * 0.035, height * 0.022, 24),
        ]
    )
    pore_count = int(rng.integers(18, 38))
    for _ in range(pore_count):
        x = rng.uniform(cx - width * 0.34, cx + width * 0.34)
        y = rng.uniform(cy - height * 0.02, cy + height * 0.37)
        structures.append(_ellipse(x, y, rng.uniform(1.2, 3.0), rng.uniform(1.0, 2.5), 18))
    context = []
    context_fills = []
    context_inlays = []
    single_individual = (
        bool(forced_single) if forced_single is not None else bool(rng.random() < 0.16)
    )
    if not single_individual:
        spacing_x = width * rng.uniform(1.05, 1.15)
        spacing_y = height * rng.uniform(1.00, 1.10)
        for row in (-1, 0, 1):
            for column in (-1, 0, 1):
                if row == 0 and column == 0:
                    continue
                nx = cx + column * spacing_x + (row % 2) * spacing_x * 0.45
                ny = cy + row * spacing_y
                neighbour_width = width * rng.uniform(0.90, 1.10)
                neighbour_height = height * rng.uniform(0.91, 1.09)
                neighbour = _zooid_outline(
                    nx,
                    ny,
                    neighbour_width,
                    neighbour_height,
                    3.5,
                    rng,
                )
                context.append(neighbour)
                context_fills.append(neighbour)
                neighbour_orifice_y = ny - neighbour_height * 0.24
                neighbour_orifice = _ellipse(
                    nx + rng.uniform(-3, 3),
                    neighbour_orifice_y,
                    neighbour_width * 0.28,
                    neighbour_height * 0.15,
                    80,
                    wobble=rng.uniform(0.01, 0.06),
                    phase=rng.uniform(0, 6),
                    harmonic=3,
                )
                context.append(neighbour_orifice)
                context_inlays.append(neighbour_orifice)
                for _ in range(int(rng.integers(14, 30))):
                    pore_x = rng.uniform(
                        nx - neighbour_width * 0.34, nx + neighbour_width * 0.34
                    )
                    pore_y = rng.uniform(
                        ny - neighbour_height * 0.02, ny + neighbour_height * 0.37
                    )
                    context.append(
                        _ellipse(
                            pore_x,
                            pore_y,
                            rng.uniform(1.2, 3.0),
                            rng.uniform(1.0, 2.5),
                            16,
                        )
                    )
    landmarks = [
        [orifice_cx, orifice_cy - orifice_ry],
        [orifice_cx - orifice_rx, orifice_cy],
        [orifice_cx + orifice_rx, orifice_cy],
        [orifice_cx, orifice_cy + orifice_ry],
        [cx - width * 0.37, cy - height * 0.33],
        [cx + width * 0.37, cy - height * 0.33],
        [cx - width * 0.49, cy],
        [cx + width * 0.49, cy],
        [cx - width * 0.34, cy + height * 0.42],
        [cx + width * 0.34, cy + height * 0.42],
        [cx - width * 0.23, cy + height * 0.18],
        [cx + width * 0.23, cy + height * 0.18],
        [orifice_cx - orifice_rx / math.sqrt(2), orifice_cy - orifice_ry / math.sqrt(2)],
        [orifice_cx + orifice_rx / math.sqrt(2), orifice_cy - orifice_ry / math.sqrt(2)],
        [orifice_cx - orifice_rx / math.sqrt(2), orifice_cy + orifice_ry / math.sqrt(2)],
        [orifice_cx + orifice_rx / math.sqrt(2), orifice_cy + orifice_ry / math.sqrt(2)],
        median_pore_left,
        median_pore_right,
    ]
    morphology = _morphology(
        [outline],
        structures,
        landmarks,
        context=context,
        context_fills=context_fills,
        context_inlays=context_inlays,
        inlays=[orifice],
        notes={
            "pore_count": pore_count,
            "colony_context": not single_individual,
            "single_individual": single_individual,
            "single_individual_probability": 0.16,
            "crop_rule": "target_complete_neighbours_partial_same_distribution",
        },
    )
    morphology["protected_structure_indices"] = [0, 1, 2]
    return morphology


def trilobite(rng, forced_full_body=None):
    cx = 128 + rng.uniform(-6, 6)
    full_body = (
        bool(forced_full_body)
        if forced_full_body is not None
        else bool(rng.random() < 0.40)
    )
    # The old fixed 67--91 range reserved almost the whole lower image for the
    # thorax, so even isolated cephala were systematically biased upward.
    # Long individuals can instead extend beyond the crop; the labelled
    # cephalon itself is checked after augmentation and must remain complete.
    ceph_y = rng.uniform(96, 132) if full_body else rng.uniform(113, 153)
    ceph_w = rng.uniform(58, 112)
    ceph_h = rng.uniform(37, 82)
    profile = str(rng.choice(["rounded", "wide", "triangular", "swept"]))
    profile_parameters = {
        "rounded": (0.62, 0.76, 0.16, 0.52),
        "wide": (0.76, 0.58, 0.06, 0.43),
        "triangular": (0.43, 0.88, 0.24, 0.58),
        "swept": (0.68, 0.66, 0.25, 0.40),
    }
    crown_x, crown_y, lateral_y, rear_fraction = profile_parameters[profile]
    # The downstream Trilomorph task is cephalon-centric; retain some whole
    # bodies for context, but make isolated cephala the majority curriculum.
    spine_base = rng.uniform(0, 40)
    left_spine = spine_base * rng.uniform(0.90, 1.10)
    right_spine = spine_base * rng.uniform(0.90, 1.10)
    centre_skew = rng.uniform(-ceph_w * 0.025, ceph_w * 0.025)
    genal_y = rng.uniform(0.50, 0.74)
    rear_y = rng.uniform(0.40, 0.57)
    centre_rear_y = rng.uniform(0.45, 0.62)
    paired_jitter = 0.025
    cephalon_controls = [
        [cx + centre_skew, ceph_y - ceph_h],
        [cx - ceph_w * crown_x, ceph_y - ceph_h * crown_y],
        [cx - ceph_w, ceph_y - ceph_h * lateral_y],
        [cx - ceph_w - left_spine, ceph_y + ceph_h * (genal_y + rng.uniform(-paired_jitter, paired_jitter))],
        [cx - ceph_w * rear_fraction, ceph_y + ceph_h * (rear_y + rng.uniform(-paired_jitter, paired_jitter))],
        [cx, ceph_y + ceph_h * centre_rear_y],
        [cx + ceph_w * rear_fraction, ceph_y + ceph_h * (rear_y + rng.uniform(-paired_jitter, paired_jitter))],
        [cx + ceph_w + right_spine, ceph_y + ceph_h * (genal_y + rng.uniform(-paired_jitter, paired_jitter))],
        [cx + ceph_w, ceph_y - ceph_h * lateral_y],
        [cx + ceph_w * crown_x, ceph_y - ceph_h * crown_y],
    ]
    cephalon = _curve(cephalon_controls, 180)
    fills = [cephalon]
    structures = []
    glab_w = ceph_w * rng.uniform(0.20, 0.39)
    glab_h = ceph_h * rng.uniform(0.66, 1.05)
    glab_cx = cx + rng.uniform(-ceph_w * 0.015, ceph_w * 0.015)
    glab_cy = ceph_y - ceph_h * 0.20
    glabella = _ellipse(
        glab_cx,
        glab_cy,
        glab_w,
        glab_h,
        100,
        wobble=rng.uniform(0.02, 0.09),
        phase=rng.uniform(0, 6),
    )
    structures.append(glabella)
    eye_x = ceph_w * rng.uniform(0.43, 0.64)
    eye_y = glab_cy + rng.uniform(-4, 7)
    eye_rx, eye_ry = rng.uniform(7, 13), rng.uniform(13, 22)
    left_eye = _ellipse(cx - eye_x, eye_y, eye_rx, eye_ry, 60, wobble=0.04)
    right_eye_cx = cx + eye_x + rng.uniform(-1.2, 1.2)
    right_eye_cy = eye_y + rng.uniform(-1.2, 1.2)
    right_eye_rx = eye_rx * rng.uniform(0.95, 1.05)
    right_eye_ry = eye_ry * rng.uniform(0.95, 1.05)
    right_eye = _ellipse(
        right_eye_cx,
        right_eye_cy,
        right_eye_rx,
        right_eye_ry,
        60,
        wobble=0.04,
    )
    structures.extend([left_eye, right_eye])
    left_suture = np.array(
        [
            [glab_cx - glab_w - 6, ceph_y - ceph_h * 0.78],
            [cx - eye_x - 3, eye_y],
            [cx - ceph_w * 0.52, ceph_y + ceph_h * 0.38],
        ]
    )
    right_suture = np.array(
        [
            [glab_cx + glab_w + 6, ceph_y - ceph_h * 0.78],
            [right_eye_cx + 3, right_eye_cy],
            [cx + ceph_w * 0.52, ceph_y + ceph_h * 0.38],
        ]
    )
    structures.extend([left_suture, right_suture])
    # Three paired glabellar furrows divide the axial lobe into visible
    # anterior, middle and posterior lobes.  Their outer endpoints are the six
    # new landmarks, while curved inner portions form biologically meaningful
    # junctions rather than arbitrary points.
    furrow_landmarks = []
    glabellar_furrows = []
    for fraction, width_fraction, sweep in (
        (-0.48, 0.78, -0.08),
        (-0.10, 0.88, 0.02),
        (0.28, 0.94, 0.10),
    ):
        y = glab_cy + glab_h * fraction
        half = glab_w * width_fraction
        left_point = np.array([glab_cx - half, y], dtype=np.float32)
        right_point = np.array([glab_cx + half, y], dtype=np.float32)
        furrow = np.array(
            [
                left_point,
                [glab_cx - glab_w * 0.34, y + glab_h * sweep],
                [glab_cx, y + glab_h * (sweep + rng.uniform(-0.025, 0.025))],
                [glab_cx + glab_w * 0.34, y + glab_h * sweep],
                right_point,
            ],
            dtype=np.float32,
        )
        glabellar_furrows.append(furrow)
        furrow_landmarks.extend([left_point, right_point])
    protected_structure_indices = list(
        range(len(structures), len(structures) + len(glabellar_furrows))
    )
    structures.extend(glabellar_furrows)

    occipital_y = glab_cy + glab_h * 0.62
    occipital_ring = _open_arc(
        glab_cx,
        occipital_y,
        glab_w * 1.04,
        glab_h * 0.22,
        math.pi * 0.08,
        math.pi * 0.92,
        42,
    )
    anterior_border = _open_arc(
        cx,
        ceph_y - ceph_h * 0.08,
        ceph_w * 0.82,
        ceph_h * 0.70,
        math.pi * 1.10,
        math.pi * 1.90,
        58,
    )
    structures.extend([occipital_ring, anterior_border])
    for side in (-1, 1):
        cheek_x = cx + side * ceph_w * 0.68
        structures.append(
            np.array(
                [
                    [cheek_x, eye_y - ceph_h * 0.18],
                    [cheek_x + side * ceph_w * 0.10, eye_y + ceph_h * 0.02],
                    [cheek_x + side * ceph_w * 0.15, eye_y + ceph_h * 0.22],
                ],
                dtype=np.float32,
            )
        )
    body_crop_intended = False
    thorax_bottom = None
    thorax = None
    thorax_segment_y = []
    if full_body:
        thorax_top = ceph_y + ceph_h * 0.46
        body_crop_intended = bool(rng.random() < 0.64)
        body_length = (
            rng.uniform(155, 235)
            if body_crop_intended
            else rng.uniform(82, 126)
        )
        thorax_bottom = thorax_top + body_length
        thorax_w = ceph_w * rng.uniform(0.60, 0.82)
        thorax = _curve(
            [
                [cx - thorax_w, thorax_top],
                [cx - thorax_w * 0.92, (thorax_top + thorax_bottom) / 2],
                [cx - thorax_w * 0.40, thorax_bottom],
                [cx, thorax_bottom + rng.uniform(8, 20)],
                [cx + thorax_w * 0.40, thorax_bottom],
                [cx + thorax_w * 0.92, (thorax_top + thorax_bottom) / 2],
                [cx + thorax_w, thorax_top],
            ],
            160,
        )
        fills.append(thorax)
        segment_count = int(rng.integers(6, 12))
        thorax_segment_y = np.linspace(
            thorax_top + 10, thorax_bottom - 7, segment_count
        )
        for y in thorax_segment_y:
            fraction = (y - thorax_top) / max(thorax_bottom - thorax_top, 1)
            half = thorax_w * (1 - 0.45 * fraction)
            structures.append(
                np.array([[cx - half, y], [cx, y + rng.uniform(1, 5)], [cx + half, y]])
            )
        structures.append(np.array([[cx, thorax_top], [cx, thorax_bottom + 8]]))
    landmarks = [
        [glab_cx, glab_cy - glab_h],
        [glab_cx, glab_cy + glab_h],
        [cx - eye_x, eye_y - eye_ry * 0.72],
        [right_eye_cx, right_eye_cy - right_eye_ry * 0.72],
        [cx - eye_x, eye_y + eye_ry * 0.72],
        [right_eye_cx, right_eye_cy + right_eye_ry * 0.72],
        cephalon_controls[3],
        cephalon_controls[7],
        left_suture[1],
        right_suture[1],
        cephalon_controls[4],
        cephalon_controls[6],
        *furrow_landmarks,
    ]
    cheek_relief = rng.uniform(5.5, 10.5)
    glabella_relief = rng.uniform(14.0, 25.0)
    eye_relief = rng.uniform(6.5, 12.5)
    furrow_depth = rng.uniform(2.4, 4.8)
    border_relief = rng.uniform(1.4, 3.8)
    surface, surface_light = _trilobite_cephalon_surface(
        cephalon,
        [glab_cx, glab_cy],
        [glab_w, glab_h],
        [[cx - eye_x, eye_y], [right_eye_cx, right_eye_cy]],
        [[eye_rx, eye_ry], [right_eye_rx, right_eye_ry]],
        glabellar_furrows,
        [left_suture, right_suture],
        rng,
        cheek_relief,
        glabella_relief,
        eye_relief,
        furrow_depth,
        border_relief,
    )
    thorax_pleural_relief = None
    thorax_axial_relief = None
    if thorax is not None:
        thorax_pleural_relief = rng.uniform(4.5, 8.5)
        thorax_axial_relief = rng.uniform(5.0, 10.0)
        thorax_surface = _trilobite_thorax_surface(
            thorax,
            thorax_segment_y,
            rng,
            thorax_pleural_relief,
            thorax_axial_relief,
            surface_light,
        )
        surface = _merge_surfaces(surface, thorax_surface)
    morphology = _morphology(
        fills,
        structures,
        landmarks,
        notes={
            "full_body": full_body,
            "cephalon_center_y": float(ceph_y),
            "cephalon_visibility_rule": "fully_visible_after_layout",
            "body_crop_intended": body_crop_intended,
            "thorax_bottom_unclipped": (
                float(thorax_bottom) if thorax_bottom is not None else None
            ),
            "cephalon_profile": profile,
            "cephalon_width": float(ceph_w),
            "cephalon_height": float(ceph_h),
            "glabellar_furrow_pairs": 3,
            "occipital_ring": True,
            "marginal_detail_lines": 3,
            "left_genal_spine": float(left_spine),
            "right_genal_spine": float(right_spine),
            "geometry": "three_dimensional_trilobite_relief_field",
            "real_reference_basis": "local_trilomorph_selected_cephala",
            "surface_height_hierarchy": "glabella_above_eyes_above_cheeks",
            "glabella_surface_relief": float(glabella_relief),
            "eye_surface_relief": float(eye_relief),
            "cheek_surface_relief": float(cheek_relief),
            "furrow_surface_depth": float(furrow_depth),
            "border_surface_relief": float(border_relief),
            "surface_components": ["cephalon"] + (["thorax"] if full_body else []),
            "thorax_pleural_relief": (
                float(thorax_pleural_relief)
                if thorax_pleural_relief is not None
                else None
            ),
            "thorax_axial_relief": (
                float(thorax_axial_relief)
                if thorax_axial_relief is not None
                else None
            ),
            "surface_light_direction": surface_light.tolist(),
        },
    )
    morphology["surface"] = surface
    morphology["protected_structure_indices"] = protected_structure_indices
    return morphology


GENERATORS = OrderedDict(
    [
        ("conodont_element", conodont_element),
        ("fusulinid_section", fusulinid_section),
        ("gastropod_shell", gastropod_shell),
        ("microtus_dentition", microtus_dentition),
        ("bryozoan_zooid", bryozoan_zooid),
        ("trilobite", trilobite),
    ]
)


def _choose_style(domain, rng, forced_style=None):
    if forced_style is not None:
        if forced_style not in STYLE_WEIGHTS[domain]:
            choices = ", ".join(STYLE_WEIGHTS[domain])
            raise ValueError(f"style {forced_style!r} invalid for {domain}; choose {choices}")
        return forced_style
    weights = STYLE_WEIGHTS[domain]
    return str(rng.choice(list(weights), p=list(weights.values())))


def _background(style, size, rng):
    yy, xx = np.mgrid[:size, :size]
    if style == "sem":
        base = rng.uniform(2, 28)
        image = base + rng.normal(0, rng.uniform(1.5, 6), (size, size))
        vignette = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size**2)
        image -= vignette * rng.uniform(4, 22)
    elif style in {"plate", "thin_section"}:
        base = rng.uniform(225, 253)
        image = base + rng.normal(0, rng.uniform(0.8, 4), (size, size))
        if rng.random() < 0.45:
            image += rng.uniform(-18, 5) * (xx / size - 0.5)
    elif style == "macro":
        base = rng.uniform(35, 128)
        image = base + rng.normal(0, rng.uniform(2, 9), (size, size))
        for _ in range(int(rng.integers(20, 55))):
            px, py = rng.uniform(0, size, 2)
            radius = rng.uniform(2, 13)
            blob = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * radius**2))
            image += blob * rng.uniform(-28, 32)
    elif style == "colony":
        base = rng.uniform(125, 206)
        image = base + rng.normal(0, rng.uniform(3, 10), (size, size))
        image += rng.uniform(-18, 18) * (yy / size - 0.5)
    else:  # matrix
        base = rng.uniform(55, 190)
        angle = rng.uniform(0, math.pi)
        along = xx * math.cos(angle) + yy * math.sin(angle)
        image = base + rng.normal(0, rng.uniform(4, 14), (size, size))
        image += rng.uniform(3, 16) * np.sin(along * rng.uniform(0.025, 0.09) + rng.uniform(0, 6))
        for _ in range(int(rng.integers(4, 14))):
            px, py = rng.uniform(0, size, 2)
            radius = rng.uniform(4, 22)
            blob = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * radius**2))
            image += blob * rng.uniform(-42, 42)
    return np.clip(image, 0, 255).astype(np.float32)


def _layout(domain, rng):
    scale_range, angle_range, shift_x, shift_y, shear_range = LAYOUT_RANGES[domain]
    return {
        "scale": rng.uniform(*scale_range),
        "scale_x": rng.uniform(0.86, 1.14),
        "scale_y": rng.uniform(0.88, 1.12),
        "angle": rng.uniform(*angle_range),
        "shift": np.array([rng.uniform(*shift_x), rng.uniform(*shift_y)]),
        "shear": rng.uniform(*shear_range),
    }


def _transform_points(points, layout):
    points = np.asarray(points, dtype=np.float32)
    centered = points - BASE_SIZE / 2
    matrix = np.array(
        [
            [layout["scale_x"], layout["shear"]],
            [0.0, layout["scale_y"]],
        ],
        dtype=np.float32,
    )
    centered = centered @ matrix.T * layout["scale"]
    angle = math.radians(layout["angle"])
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    return centered @ rotation.T + BASE_SIZE / 2 + layout["shift"]


def _draw_polygon(mask, polygon, fill=255):
    ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon], fill=fill)


def _artifact_font(size, serif=False):
    candidates = (
        ("DejaVuSerif.ttf", "DejaVuSans.ttf")
        if serif
        else ("DejaVuSans.ttf", "DejaVuSerif.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _boxes_overlap(first, second, margin=0):
    return not (
        first[2] + margin < second[0]
        or second[2] + margin < first[0]
        or first[3] + margin < second[1]
        or second[3] + margin < first[1]
    )


def _artifact_position(width, height, mask, landmarks, occupied, rng):
    """Place publication furniture away from target landmarks and dense mask."""
    render_size = mask.shape[0]
    margin = int(rng.integers(10, 25))
    xs = [margin, render_size - width - margin, (render_size - width) // 2]
    ys = [margin, render_size - height - margin, (render_size - height) // 2]
    candidates = [
        (xs[0], ys[0]),
        (xs[1], ys[0]),
        (xs[0], ys[1]),
        (xs[1], ys[1]),
        (xs[0], ys[2]),
        (xs[1], ys[2]),
        (xs[2], ys[0]),
        (xs[2], ys[1]),
    ]
    candidates = [
        (
            int(np.clip(x + rng.integers(-10, 11), 4, render_size - width - 4)),
            int(np.clip(y + rng.integers(-10, 11), 4, render_size - height - 4)),
        )
        for x, y in candidates
    ]
    order = rng.permutation(len(candidates))
    for maximum_occupancy in (0.18, 0.42):
        for index in order:
            x, y = candidates[int(index)]
            box = (x, y, x + width, y + height)
            expanded = (
                box[0] - 18,
                box[1] - 18,
                box[2] + 18,
                box[3] + 18,
            )
            if any(_boxes_overlap(box, other, margin=12) for other in occupied):
                continue
            if any(
                expanded[0] <= point[0] <= expanded[2]
                and expanded[1] <= point[1] <= expanded[3]
                for point in landmarks
            ):
                continue
            crop = mask[box[1] : box[3] + 1, box[0] : box[2] + 1]
            if crop.size and float(np.mean(crop > 0)) <= maximum_occupancy:
                return box
    return None


def _draw_acquisition_artifacts(
    image, mask_image, landmarks_highres, rng, forced_count=None
):
    """Draw sparse scale/plate furniture that never changes the target mask."""
    mask = np.asarray(mask_image, dtype=np.uint8)
    records = []
    occupied = []
    count = (
        int(forced_count)
        if forced_count is not None
        else 1 + int(rng.random() < SECOND_ACQUISITION_ARTIFACT_PROBABILITY)
    )
    for _ in range(count):
        artifact_class = str(
            rng.choice(
                ["scale_bar", "plate_index", "figure_label", "specimen_id"],
                p=[0.52, 0.20, 0.17, 0.11],
            )
        )
        draw = ImageDraw.Draw(image)
        if artifact_class == "scale_bar":
            bar_style = str(
                rng.choice(["solid", "end_ticks", "segmented", "ruler_ticks"])
            )
            label = (
                str(rng.choice(["1 mm", "2 mm", "5 mm", "10 mm", "0.5 mm", "500 µm"]))
                if rng.random() < 0.76
                else None
            )
            font = _artifact_font(int(rng.integers(18, 29)), serif=rng.random() < 0.30)
            bar_width = int(rng.integers(58, 151))
            bar_height = int(rng.integers(5, 11))
            text_box = draw.textbbox((0, 0), label, font=font) if label else (0, 0, 0, 0)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            width = max(bar_width, text_width) + 8
            height = bar_height + (text_height + 8 if label else 8)
            box = _artifact_position(
                width, height, mask, landmarks_highres, occupied, rng
            )
            if box is None:
                continue
            x, y = box[0], box[1]
            crop = np.asarray(image, dtype=np.uint8)[box[1] : box[3], box[0] : box[2]]
            local_level = float(np.median(crop)) if crop.size else 128.0
            tone = 244 if local_level < 128 else 10
            muted = int(np.clip(local_level + (45 if tone > local_level else -45), 5, 250))
            text_y = y
            bar_y = y + (text_height + 6 if label else 3)
            bar_x = x + (width - bar_width) // 2
            if label:
                draw.text(
                    (x + (width - text_width) // 2, text_y),
                    label,
                    fill=tone,
                    font=font,
                    stroke_width=int(rng.random() < 0.18),
                    stroke_fill=tone,
                )
            if bar_style == "solid":
                draw.rectangle(
                    (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
                    fill=tone,
                )
            elif bar_style == "end_ticks":
                centre_y = bar_y + bar_height // 2
                draw.line((bar_x, centre_y, bar_x + bar_width, centre_y), fill=tone, width=4)
                tick = max(8, bar_height + 5)
                draw.line((bar_x, centre_y - tick, bar_x, centre_y + tick), fill=tone, width=3)
                draw.line((bar_x + bar_width, centre_y - tick, bar_x + bar_width, centre_y + tick), fill=tone, width=3)
            elif bar_style == "segmented":
                segments = int(rng.choice([4, 5, 6]))
                for segment in range(segments):
                    left = bar_x + round(bar_width * segment / segments)
                    right = bar_x + round(bar_width * (segment + 1) / segments)
                    draw.rectangle(
                        (left, bar_y, right, bar_y + bar_height),
                        fill=tone if segment % 2 == 0 else muted,
                    )
                draw.rectangle(
                    (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height),
                    outline=tone,
                    width=2,
                )
            else:
                centre_y = bar_y + bar_height // 2
                draw.line((bar_x, centre_y, bar_x + bar_width, centre_y), fill=tone, width=3)
                for tick_index in range(6):
                    tick_x = bar_x + round(bar_width * tick_index / 5)
                    tick_height = bar_height + (5 if tick_index in {0, 5} else 1)
                    draw.line(
                        (tick_x, centre_y - tick_height, tick_x, centre_y + 2),
                        fill=tone,
                        width=2,
                    )
            record = {
                "type": artifact_class,
                "style": bar_style,
                "label": label,
                "bbox": [float(value / RENDER_SCALE) for value in box],
            }
        else:
            if artifact_class == "plate_index":
                label = str(rng.choice(list("ABCDEF") + ["1", "2", "3", "4"]))
                font_size = int(rng.integers(32, 53))
            elif artifact_class == "figure_label":
                label = str(
                    rng.choice(
                        [
                            f"Fig. {int(rng.integers(1, 13))}{str(rng.choice(list('ABCDEF')))}",
                            f"Pl. {int(rng.integers(1, 9))}",
                            f"No. {int(rng.integers(1, 40))}",
                            f"{int(rng.integers(1, 10))}{str(rng.choice(list('abcd')))}",
                        ]
                    )
                )
                font_size = int(rng.integers(19, 30))
            else:
                label = str(
                    rng.choice(
                        [
                            f"MB.T.{int(rng.integers(1000, 9999))}",
                            f"NUF {int(rng.integers(1000, 9999))}",
                            f"USNM {int(rng.integers(10000, 999999))}",
                            f"Spec. {int(rng.integers(1, 999))}",
                        ]
                    )
                )
                font_size = int(rng.integers(17, 26))
            font = _artifact_font(font_size, serif=rng.random() < 0.48)
            text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
            width = text_box[2] - text_box[0] + 12
            height = text_box[3] - text_box[1] + 12
            box = _artifact_position(
                width, height, mask, landmarks_highres, occupied, rng
            )
            if box is None:
                continue
            crop = np.asarray(image, dtype=np.uint8)[box[1] : box[3], box[0] : box[2]]
            local_level = float(np.median(crop)) if crop.size else 128.0
            tone = 246 if local_level < 128 else 8
            backing = bool(rng.random() < 0.13)
            if backing:
                backing_tone = 12 if tone > 128 else 244
                draw.rounded_rectangle(box, radius=3, fill=backing_tone)
            draw.text(
                (box[0] + 6, box[1] + 4),
                label,
                fill=tone,
                font=font,
                stroke_width=int(rng.random() < 0.22),
                stroke_fill=tone,
            )
            record = {
                "type": artifact_class,
                "style": "backed_text" if backing else "plain_text",
                "label": label,
                "bbox": [float(value / RENDER_SCALE) for value in box],
            }
        occupied.append(box)
        records.append(record)
    return image, records


def _render(domain, morphology, style, rng):
    render_size = BASE_SIZE * RENDER_SCALE
    layout = _layout(domain, rng)
    fills = [_transform_points(item, layout) * RENDER_SCALE for item in morphology["fills"]]
    structures = [
        _transform_points(item, layout) * RENDER_SCALE for item in morphology["structures"]
    ]
    context = [_transform_points(item, layout) * RENDER_SCALE for item in morphology["context"]]
    context_fills = [
        _transform_points(item, layout) * RENDER_SCALE
        for item in morphology.get("context_fills", ())
    ]
    context_inlays = [
        _transform_points(item, layout) * RENDER_SCALE
        for item in morphology.get("context_inlays", ())
    ]
    inlays = [
        _transform_points(item, layout) * RENDER_SCALE
        for item in morphology.get("inlays", ())
    ]
    surface = None
    if "surface" in morphology:
        surface = {
            "vertices": _transform_points(
                morphology["surface"]["vertices"], layout
            )
            * RENDER_SCALE,
            "faces": morphology["surface"]["faces"],
            "depths": morphology["surface"]["depths"],
            "shades": morphology["surface"]["shades"],
        }
    landmarks = _transform_points(morphology["landmarks"], layout)

    cephalon_fully_visible = None
    if domain == "trilobite":
        cephalon = fills[0]
        visibility_margin = 2 * RENDER_SCALE
        cephalon_fully_visible = bool(
            (cephalon[:, 0] >= visibility_margin).all()
            and (cephalon[:, 0] <= render_size - visibility_margin).all()
            and (cephalon[:, 1] >= visibility_margin).all()
            and (cephalon[:, 1] <= render_size - visibility_margin).all()
        )
        if not cephalon_fully_visible:
            return None

    mask_image = Image.new("L", (render_size, render_size), 0)
    for polygon in fills:
        _draw_polygon(mask_image, polygon)
    edge_loss_records = []
    edge_loss_probabilities = {
        "conodont_element": 0.34,
        "gastropod_shell": 0.30,
        "microtus_dentition": 0.28,
    }
    if rng.random() < edge_loss_probabilities.get(domain, 0.0):
        # Remove one or occasionally two small edge fragments.  Candidate
        # centres lie on the transformed silhouette and remain well separated
        # from every landmark, so preservation noise never destroys the
        # supervisory definition itself.
        boundary = fills[0]
        protected_landmarks = landmarks * RENDER_SCALE
        candidates = boundary[
            np.min(
                np.linalg.norm(
                    boundary[:, None, :] - protected_landmarks[None, :, :], axis=2
                ),
                axis=1,
            )
            > 26
        ]
        if len(candidates):
            edge_draw = ImageDraw.Draw(mask_image)
            for _ in range(1 + int(rng.random() < 0.18)):
                centre = candidates[int(rng.integers(0, len(candidates)))]
                radius_x = rng.uniform(7, 17)
                radius_y = rng.uniform(6, 15)
                edge_draw.ellipse(
                    (
                        centre[0] - radius_x,
                        centre[1] - radius_y,
                        centre[0] + radius_x,
                        centre[1] + radius_y,
                    ),
                    fill=0,
                )
                edge_loss_records.append(
                    {
                        "center_highres": centre.tolist(),
                        "radius_highres": [float(radius_x), float(radius_y)],
                    }
                )
    mask = np.asarray(mask_image, dtype=np.uint8)
    if not np.any(mask):
        raise ValueError(f"empty transformed specimen for {domain}")

    background = _background(style, render_size, rng)
    bg_level = float(np.median(background))

    # Scans extracted from publications often contain a narrow photograph with
    # black letterbox margins.  Reproduce it sparingly so the model learns the
    # acquisition artifact without making it the defining signal.
    if domain in {"conodont_element", "fusulinid_section", "trilobite"} and rng.random() < 0.22:
        margin = int(rng.uniform(0.05, 0.15) * render_size)
        if rng.random() < 0.72:
            background[:, :margin] = rng.uniform(0, 10)
            background[:, -margin:] = rng.uniform(0, 10)
        else:
            background[: margin // 2] = rng.uniform(0, 10)
            background[-margin // 2 :] = rng.uniform(0, 10)

    if domain == "bryozoan_zooid":
        specimen_level = bg_level
    elif style == "sem":
        specimen_level = rng.uniform(118, 226)
    elif style in {"plate", "thin_section"}:
        specimen_level = rng.uniform(38, 172)
    elif style == "macro":
        specimen_level = np.clip(
            bg_level + rng.choice([-1, 1]) * rng.uniform(24, 78), 28, 230
        )
    elif style == "colony":
        specimen_level = np.clip(bg_level + rng.uniform(-38, 28), 45, 225)
    else:
        specimen_level = np.clip(
            bg_level + rng.choice([-1, 1]) * rng.uniform(28, 92), 28, 232
        )
    ink_direction = -1 if specimen_level > bg_level else 1
    if domain == "bryozoan_zooid":
        ink_direction = -1 if bg_level >= 128 else 1

    # Neighbouring zooids, substrate grains and soft-body context sit behind
    # the labelled specimen.  Drawing them into the background prevents them
    # from erasing the target when shapes overlap.
    context_layer = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    context_draw = ImageDraw.Draw(context_layer)
    if domain == "bryozoan_zooid":
        context_fill_level = specimen_level + rng.uniform(-4, 4)
        context_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(42, 68), 8, 247)
        )
    else:
        context_fill_level = bg_level + rng.uniform(-18, 18)
        context_tone = int(
            np.clip(bg_level + rng.choice([-1, 1]) * rng.uniform(8, 30), 8, 247)
        )
    for polygon in context_fills:
        if domain != "bryozoan_zooid":
            context_draw.polygon(
                [tuple(point) for point in polygon],
                fill=int(
                    np.clip(context_fill_level + rng.uniform(-3, 3), 8, 247)
                ),
            )
    for polygon in context_inlays:
        context_draw.polygon(
            [tuple(point) for point in polygon],
            fill=int(
                np.clip(
                    specimen_level + ink_direction * rng.uniform(50, 78), 5, 250
                )
            ),
        )
    for line in context:
        context_draw.line(
            [tuple(point) for point in np.vstack((line, line[0]))],
            fill=context_tone,
            width=int(rng.integers(5, 9)) if domain == "bryozoan_zooid" else int(rng.integers(3, 8)),
        )
    background = np.asarray(context_layer, dtype=np.float32)
    yy, xx = np.mgrid[:render_size, :render_size]
    if domain == "bryozoan_zooid":
        # The labelled zooid is not a separately coloured foreground object;
        # it samples the same continuous colony texture as its neighbours.
        texture = background + rng.normal(0, rng.uniform(0.8, 2.2), background.shape)
    else:
        texture = specimen_level + rng.normal(0, rng.uniform(3, 15), (render_size, render_size))
        texture += rng.uniform(3, 14) * np.sin(xx * rng.uniform(0.018, 0.07) + rng.uniform(0, 6))
        texture += rng.uniform(2, 10) * np.cos(yy * rng.uniform(0.018, 0.08) + rng.uniform(0, 6))
        light_x, light_y = rng.uniform(-0.45, 0.45, 2)
        texture *= np.clip(
            1 + light_x * (xx / render_size - 0.5) + light_y * (yy / render_size - 0.5),
            0.55,
            1.45,
        )
    image = np.where(mask > 0, texture, background)
    image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))

    if surface is not None:
        surface_layer = image.copy()
        surface_draw = ImageDraw.Draw(surface_layer)
        order = np.argsort(surface["depths"])
        surface_contrast_ranges = {
            "conodont_element": (52, 84),
            "gastropod_shell": (72, 112),
            "microtus_dentition": (62, 102),
            "trilobite": (62, 104),
        }
        surface_contrast = rng.uniform(*surface_contrast_ranges.get(domain, (58, 96)))
        for face_index in order:
            polygon = surface["vertices"][surface["faces"][face_index]]
            tone = specimen_level + (surface["shades"][face_index] - 0.46) * surface_contrast
            tone += rng.normal(0, 2.5)
            surface_draw.polygon(
                [tuple(point) for point in polygon],
                fill=int(np.clip(tone, 4, 251)),
            )
        image = Image.composite(surface_layer, image, mask_image)

    if rng.random() < 0.42:
        damage = Image.new("L", (render_size, render_size), 0)
        draw = ImageDraw.Draw(damage)
        for _ in range(int(rng.integers(1, 6))):
            x, y = rng.uniform(0, render_size, 2)
            rx, ry = rng.uniform(5, 25), rng.uniform(3, 18)
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=int(rng.integers(80, 220)))
        damage_array = np.asarray(damage, dtype=np.float32) / 255
        damaged = np.asarray(image, dtype=np.float32)
        damaged = damaged * (1 - damage_array * rng.uniform(0.20, 0.65)) + background * damage_array * rng.uniform(0.20, 0.65)
        image = Image.fromarray(np.clip(damaged, 0, 255).astype(np.uint8))

    if domain == "bryozoan_zooid":
        outline_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(42, 68), 8, 247)
        )
        structure_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(30, 52), 8, 247)
        )
    elif domain == "gastropod_shell":
        outline_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(14, 35), 8, 247)
        )
        structure_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(10, 28), 8, 247)
        )
    elif domain == "trilobite":
        # Relief carries most of the anatomy.  These restrained lines retain
        # sutures/furrows in eroded casts without flattening the domes into a
        # diagrammatic ink drawing.
        outline_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(18, 46), 8, 247)
        )
        structure_tone = int(
            np.clip(specimen_level + ink_direction * rng.uniform(13, 38), 8, 247)
        )
    else:
        outline_tone = int(np.clip(specimen_level + ink_direction * rng.uniform(28, 78), 8, 247))
        structure_tone = int(np.clip(specimen_level + ink_direction * rng.uniform(22, 68), 8, 247))
    outline_layer = image.copy()
    outline_draw = ImageDraw.Draw(outline_layer)
    for polygon in fills:
        outline_width = (
            int(rng.choice([5, 6, 7, 8]))
            if domain == "bryozoan_zooid"
            else int(rng.choice([2, 3, 4, 5]))
        )
        outline_draw.line(
            [tuple(point) for point in np.vstack((polygon, polygon[0]))],
            fill=outline_tone,
            width=outline_width,
        )
    image = Image.composite(outline_layer, image, mask_image)

    if inlays:
        inlay_layer = image.copy()
        inlay_draw = ImageDraw.Draw(inlay_layer)
        if domain == "microtus_dentition":
            inlay_range = (20, 48)
        elif domain == "gastropod_shell":
            inlay_range = (28, 58)
        elif domain == "bryozoan_zooid":
            inlay_range = (50, 78)
        else:
            inlay_range = (50, 105)
        for polygon in inlays:
            inlay_draw.polygon(
                [tuple(point) for point in polygon],
                fill=int(
                    np.clip(
                        specimen_level + ink_direction * rng.uniform(*inlay_range),
                        5,
                        250,
                    )
                ),
            )
        image = Image.composite(inlay_layer, image, mask_image)

    structure_layer = image.copy()
    structure_draw = ImageDraw.Draw(structure_layer)
    structure_kinds = morphology.get("structure_kinds", ["generic"] * len(structures))
    protected_structure_indices = set(
        morphology.get("protected_structure_indices", ())
    )
    structure_source = np.asarray(image, dtype=np.uint8)
    conodont_light = np.asarray(
        morphology.get("notes", {}).get("surface_light_direction", [-0.4, -0.4, 0.8])[:2],
        dtype=np.float32,
    )
    light_norm = max(float(np.linalg.norm(conodont_light)), 1e-6)
    conodont_light /= light_norm
    for structure_index, line in enumerate(structures):
        # Every conodont landmark is a rendered ridge/axis junction, so those
        # supporting lines are protected from stochastic structure dropout.
        if (
            domain != "conodont_element"
            and structure_index not in protected_structure_indices
            and rng.random() < 0.08
        ):
            continue
        if domain == "conodont_element":
            # A fixed global ink value made these lines appear pasted over the
            # curved blade.  Each short segment now follows its underlying
            # surface tone and differs only enough to read as a shallow ridge.
            kind = structure_kinds[structure_index]
            for first, second in zip(line[:-1], line[1:]):
                midpoint = (first + second) / 2
                mx = int(np.clip(round(float(midpoint[0])), 0, render_size - 1))
                my = int(np.clip(round(float(midpoint[1])), 0, render_size - 1))
                local_level = float(structure_source[my, mx])
                tangent = second - first
                tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
                normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
                if float(np.dot(normal, conodont_light)) < 0:
                    normal *= -1
                if kind == "transverse_ridge":
                    offset = normal * rng.uniform(1.15, 1.75)
                    highlight = int(
                        np.clip(local_level + rng.uniform(24, 39), 6, 251)
                    )
                    shadow = int(
                        np.clip(local_level - rng.uniform(20, 34), 4, 249)
                    )
                    structure_draw.line(
                        [tuple(first + offset), tuple(second + offset)],
                        fill=highlight,
                        width=2,
                    )
                    structure_draw.line(
                        [tuple(first - offset), tuple(second - offset)],
                        fill=shadow,
                        width=2,
                    )
                    # A low-contrast centre prevents a conspicuous empty gap
                    # while the paired edges carry the relief cue.
                    centre_tone = int(
                        np.clip(local_level + ink_direction * rng.uniform(5, 10), 5, 250)
                    )
                    structure_draw.line(
                        [tuple(first), tuple(second)], fill=centre_tone, width=1
                    )
                else:
                    # Longitudinal axes remain subordinate grooves; their
                    # junctions are visible but do not form a second grid.
                    groove = int(
                        np.clip(local_level + ink_direction * rng.uniform(9, 16), 5, 250)
                    )
                    structure_draw.line(
                        [tuple(first), tuple(second)], fill=groove, width=1
                    )
            continue
        if domain == "gastropod_shell":
            line_width = int(rng.choice([1, 2, 2, 3]))
        elif domain == "trilobite":
            line_width = int(rng.choice([1, 2, 2, 2, 3]))
        else:
            line_width = int(rng.choice([2, 2, 3, 3, 4]))
        structure_draw.line(
            [tuple(point) for point in line],
            fill=int(np.clip(structure_tone + rng.normal(0, 9), 5, 250)),
            width=line_width,
        )
    image = Image.composite(structure_layer, image, mask_image)

    interference_count = 0
    interference_probabilities = {
        "fusulinid_section": 0.30,
        "bryozoan_zooid": 0.22,
        "trilobite": 0.26,
        "conodont_element": 0.12,
        "gastropod_shell": 0.14,
        "microtus_dentition": 0.12,
    }
    if rng.random() < interference_probabilities[domain]:
        interference_layer = image.copy()
        interference_draw = ImageDraw.Draw(interference_layer)
        interference_count = int(rng.integers(1, 4))
        for _ in range(interference_count):
            x, y = rng.uniform(0, render_size, 2)
            length = rng.uniform(14, 48)
            angle = rng.uniform(0, 2 * math.pi)
            bend = rng.uniform(-6, 6)
            local_x = int(np.clip(round(x), 0, render_size - 1))
            local_y = int(np.clip(round(y), 0, render_size - 1))
            local_level = float(np.asarray(image, dtype=np.uint8)[local_y, local_x])
            tone = int(
                np.clip(
                    local_level + rng.choice([-1, 1]) * rng.uniform(12, 30),
                    5,
                    250,
                )
            )
            interference_draw.line(
                [
                    (x, y),
                    (
                        x + length * 0.52 * math.cos(angle) + bend,
                        y + length * 0.52 * math.sin(angle),
                    ),
                    (x + length * math.cos(angle), y + length * math.sin(angle)),
                ],
                fill=tone,
                width=int(rng.choice([1, 2, 2, 3])),
            )
        image = Image.composite(interference_layer, image, mask_image)

    # Thin-section chambers are irregularly mineralized rather than a clean
    # polar grid.  Add small light/dark chamber deposits inside the test.
    if domain == "fusulinid_section":
        deposit_layer = image.copy()
        deposit_draw = ImageDraw.Draw(deposit_layer)
        for _ in range(int(rng.integers(35, 75))):
            x, y = rng.uniform(0, render_size, 2)
            rx, ry = rng.uniform(2, 8), rng.uniform(1, 5)
            tone = int(np.clip(specimen_level + rng.choice([-1, 1]) * rng.uniform(25, 85), 5, 250))
            deposit_draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=tone)
        image = Image.composite(deposit_layer, image, mask_image)

    if rng.random() < 0.38:
        scratch_layer = image.copy()
        scratch_draw = ImageDraw.Draw(scratch_layer)
        for _ in range(int(rng.integers(1, 6))):
            x, y = rng.uniform(0, render_size, 2)
            length = rng.uniform(18, 90)
            angle = rng.uniform(0, 2 * math.pi)
            scratch_draw.line(
                (x, y, x + length * math.cos(angle), y + length * math.sin(angle)),
                fill=int(rng.uniform(25, 215)),
                width=int(rng.choice([1, 2, 3])),
            )
        image = scratch_layer

    acquisition_artifact_records = []
    artifact_probability = ACQUISITION_ARTIFACT_PROBABILITIES[style]
    if rng.random() < artifact_probability:
        image, acquisition_artifact_records = _draw_acquisition_artifacts(
            image,
            mask_image,
            landmarks * RENDER_SCALE,
            rng,
        )

    blur_ranges = {
        "sem": (0.15, 0.75),
        "plate": (0.15, 1.15),
        "thin_section": (0.10, 0.65),
        "macro": (0.35, 2.8),
        "colony": (0.20, 1.20),
        "matrix": (0.20, 1.30),
    }
    blur = rng.uniform(*blur_ranges[style])
    image = image.filter(ImageFilter.GaussianBlur(blur))
    if rng.random() < 0.28:
        image = image.filter(ImageFilter.UnsharpMask(radius=rng.uniform(0.6, 1.8), percent=int(rng.integers(60, 170))))

    image = image.resize((BASE_SIZE, BASE_SIZE), Image.Resampling.LANCZOS)
    mask_out = mask_image.resize((BASE_SIZE, BASE_SIZE), Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.uint8)
    mask_array = (np.asarray(mask_out) >= 96).astype(np.uint8)
    valid = (
        (landmarks[:, 0] >= 1)
        & (landmarks[:, 0] <= BASE_SIZE - 2)
        & (landmarks[:, 1] >= 1)
        & (landmarks[:, 1] <= BASE_SIZE - 2)
    )
    if not valid.all():
        return None
    ys, xs = np.where(mask_array > 0)
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    metadata = {
        "domain": domain,
        "style": style,
        "layout": {
            key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in layout.items()
        },
        "blur_radius_highres": float(blur),
        **morphology["notes"],
        "edge_loss_count": len(edge_loss_records),
        "edge_loss": edge_loss_records,
        "interference_feature_count": interference_count,
        "acquisition_artifact_count": len(acquisition_artifact_records),
        "acquisition_artifacts": acquisition_artifact_records,
    }
    if cephalon_fully_visible is not None:
        metadata["cephalon_fully_visible_after_layout"] = cephalon_fully_visible
    return image_array, mask_array, landmarks.astype(np.float32), bbox, metadata


def make_sample(domain, rng, forced_style=None, forced_variant=None, max_attempts=64):
    if domain not in GENERATORS:
        raise ValueError(f"unknown domain {domain!r}; choose from {list(GENERATORS)}")
    for _ in range(max_attempts):
        if domain == "bryozoan_zooid" and forced_variant is not None:
            if forced_variant not in {"single", "colony"}:
                raise ValueError("bryozoan forced_variant must be 'single' or 'colony'")
            morphology = bryozoan_zooid(
                rng, forced_single=(forced_variant == "single")
            )
        else:
            morphology = GENERATORS[domain](rng)
        style = _choose_style(domain, rng, forced_style)
        sample = _render(domain, morphology, style, rng)
        if sample is not None:
            return sample
    raise RuntimeError(f"could not place a valid {domain} sample after {max_attempts} attempts")


def resize_sample(sample, image_size):
    image, mask, keypoints, bbox, metadata = sample
    if image_size == BASE_SIZE:
        return sample
    scale = image_size / BASE_SIZE
    resized_image = np.asarray(
        Image.fromarray(image).resize((image_size, image_size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    resized_mask = (
        np.asarray(
            Image.fromarray(mask * 255).resize(
                (image_size, image_size), Image.Resampling.NEAREST
            )
        )
        > 0
    ).astype(np.uint8)
    return (
        resized_image,
        resized_mask,
        keypoints * scale,
        bbox * scale,
        metadata,
    )


def generator_manifest(cfg):
    if int(cfg.get("num_landmarks", 18)) != 18:
        raise ValueError("Synthetic requires num_landmarks: 18")
    domains = list(cfg.get("source_domains", DOMAINS))
    return {
        "generator": "paleolandmarker.data.synthetic.generate",
        "version": VERSION,
        "seed": cfg["seed"],
        "image_size": cfg["image_size"],
        "domains": domains,
        "style_weights": {domain: STYLE_WEIGHTS[domain] for domain in domains},
        "acquisition_artifact_probabilities": ACQUISITION_ARTIFACT_PROBABILITIES,
        "second_acquisition_artifact_probability": SECOND_ACQUISITION_ARTIFACT_PROBABILITY,
        "layout_ranges": {domain: LAYOUT_RANGES[domain] for domain in domains},
        "revision_notes": REVISION_NOTES,
        "counts": cfg["source_counts"],
    }


def dataset_is_current(cfg):
    """Check both the generation identity and every materialized split."""
    root = Path(cfg["data_dir"])
    manifest_path = root / "generator_manifest.json"
    try:
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_manifest = json.loads(json.dumps(generator_manifest(cfg)))
    if actual_manifest != expected_manifest:
        return False
    expected_shape = (int(cfg["image_size"]), int(cfg["image_size"]))
    for split, count in cfg["source_counts"].items():
        try:
            with np.load(root / f"source_{split}.npz", mmap_mode="r") as data:
                if len(data["images"]) != int(count):
                    return False
                if data["images"].shape[-2:] != expected_shape:
                    return False
                if data["keypoints"].shape[1] != 18:
                    return False
        except (OSError, KeyError, ValueError):
            return False
    return True


def ensure_dataset(cfg):
    """Generate once per complete synthetic-data identity, otherwise reuse it."""
    if dataset_is_current(cfg):
        print(f"reuse Synthetic data seed={cfg['seed']}: {cfg['data_dir']}")
        return False
    generate(cfg)
    return True


def generate(cfg):
    """Generate source NPZ splits using the v1-compatible dataset contract."""
    seed_everything(cfg["seed"])
    root = Path(cfg["data_dir"])
    root.mkdir(parents=True, exist_ok=True)
    domains = list(cfg.get("source_domains", DOMAINS))
    if cfg.get("num_landmarks", 18) != 18:
        raise ValueError("Synthetic requires num_landmarks: 18")
    unknown = sorted(set(domains).difference(GENERATORS))
    if unknown:
        raise ValueError(f"unknown source domains: {unknown}")

    manifest = generator_manifest(cfg)
    for split, count in cfg["source_counts"].items():
        # Keep this seed namespace byte-for-byte compatible with the paper run
        # while avoiding a legacy version label in the public source tree.
        rng = np.random.default_rng(
            cfg["seed"] + sum(map(ord, f"{chr(118)}{chr(50)}-source-{split}"))
        )
        images, masks, keypoints, boxes, labels, sample_domains, metadata = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for index in range(count):
            domain = domains[index % len(domains)]
            sample = resize_sample(make_sample(domain, rng), cfg["image_size"])
            image, mask, points, bbox, meta = sample
            images.append(image)
            masks.append(mask)
            keypoints.append(points)
            boxes.append(bbox)
            labels.append(domains.index(domain))
            sample_domains.append(domain)
            metadata.append(json.dumps(meta, sort_keys=True))
        destination = root / f"source_{split}.npz"
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}.tmp.npz"
        )
        np.savez_compressed(
            temporary,
            images=np.stack(images),
            masks=np.stack(masks),
            keypoints=np.stack(keypoints),
            visibility=np.ones((count, 18), dtype=np.uint8),
            bbox=np.stack(boxes),
            labels=np.asarray(labels),
            domains=np.asarray(sample_domains),
            landmark_names=np.asarray([LANDMARK_NAMES[domain] for domain in sample_domains]),
            sample_metadata=np.asarray(metadata),
        )
        os.replace(temporary, destination)
        print(f"saved Synthetic {destination}: {count} samples")
    manifest_path = root / "generator_manifest.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
