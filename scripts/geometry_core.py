"""Geometry only. Units: solar radii, degrees, arcsec as explicitly named."""

import numpy as np
from scipy.optimize import brentq

R_MM = 696.0
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


def unit(lon, lat):
    lon, lat = np.deg2rad([lon, lat])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def observer(lon, lat, distance_mm):
    n = unit(lon, lat)
    lon = np.deg2rad(lon)
    west = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(n, west)
    return {"n": n, "west": west, "north": north, "xyz": distance_mm * n}


def unpack(p):
    lon, lat, mu, a, b = p
    d = mu + a * np.sqrt(1 - (1 - mu * mu) / (b * b))
    e = unit(lon, lat)
    Q = np.eye(3) / (b * b) + (1 / (a * a) - 1 / (b * b)) * np.outer(e, e)
    return d, a, b, e, Q


def g1_xy(theta):
    radius = 953.35
    r0 = np.array([np.sqrt(1 - (-150 / radius) ** 2), -150 / radius, 0.0])
    endpoint = np.array([0.0, -750 / radius, np.sqrt(1 - (-750 / radius) ** 2)])
    tangent = endpoint - np.dot(endpoint, r0) * r0
    tangent /= np.linalg.norm(tangent)
    theta = np.deg2rad(np.asarray(theta))
    return (
        radius
        * (np.cos(theta)[..., None] * r0 + np.sin(theta)[..., None] * tangent)[..., :2]
    )


def point_on_ray(tx, ty, height_mm, obs):
    tx = np.asarray(tx) / ARCSEC_PER_RAD
    ty = np.asarray(ty) / ARCSEC_PER_RAD
    ray = (
        (-np.cos(ty) * np.cos(tx))[..., None] * obs["n"]
        + (np.cos(ty) * np.sin(tx))[..., None] * obs["west"]
        + np.sin(ty)[..., None] * obs["north"]
    )
    dot = ray @ obs["xyz"]
    disc = dot * dot - (obs["xyz"] @ obs["xyz"] - (R_MM + height_mm) ** 2)
    if np.any(disc < 0):
        raise ValueError(
            "A selected line of sight does not intersect the assumed height sphere."
        )
    return (obs["xyz"] + (-dot - np.sqrt(disc))[..., None] * ray) / R_MM


def project(r, obs):
    v = np.asarray(r) * R_MM - obs["xyz"]
    z = -v @ obs["n"]
    x = v @ obs["west"]
    y = v @ obs["north"]
    return ARCSEC_PER_RAD * np.stack(
        [np.arctan2(x, z), np.arctan2(y, np.sqrt(x * x + z * z))], axis=-1
    )


def field(r, p):
    d, a, b, e, Q = unpack(p)
    delta = r - d * e
    return np.sum((delta @ Q) * delta, axis=-1) - 1


def intersections(p, height_mm, obs):
    theta = np.linspace(8, 89.9, 600)
    xy = g1_xy(theta)
    v = field(point_on_ray(xy[:, 0], xy[:, 1], height_mm, obs), p)
    roots = []
    for k in range(len(theta) - 1):
        if v[k] * v[k + 1] < 0:
            roots.append(
                brentq(
                    lambda t: field(point_on_ray(*g1_xy(t), height_mm, obs), p),
                    theta[k],
                    theta[k + 1],
                    xtol=1e-10,
                )
            )
    return roots


def normal_record(p, theta, height_mm, obs):
    xy = g1_xy(theta)
    r = point_on_ray(*xy, height_mm, obs)
    d, a, b, e, Q = unpack(p)
    n = Q @ (r - d * e)
    n /= np.linalg.norm(n)
    beta = np.rad2deg(np.arcsin(np.clip(n @ obs["n"], -1, 1)))
    alpha = np.rad2deg(np.arcsin(np.clip(n @ (r / np.linalg.norm(r)), -1, 1)))
    # Finite-perspective image-plane differential, not simply orthographic n_x,n_y.
    step = 1e-5
    image_n = (project(r + step * n, obs) - project(r - step * n, obs)) / (2 * step)
    image_n /= np.linalg.norm(image_n)
    tangent = g1_xy(theta + 1e-4) - g1_xy(theta - 1e-4)
    tangent /= np.linalg.norm(tangent)
    misalignment = np.rad2deg(np.arccos(np.clip(image_n @ tangent, -1, 1)))
    return {
        "assumed_height_Mm": height_mm,
        "g1_theta_deg": theta,
        "x_arcsec": float(xy[0]),
        "y_arcsec": float(xy[1]),
        "inclination_to_sky_signed_deg": float(beta),
        "inclination_to_sky_abs_deg": float(abs(beta)),
        "inclination_above_local_tangent_signed_deg": float(alpha),
        "normal_to_G1_projected_angle_deg": float(misalignment),
        "position_HGS_Rsun": r.tolist(),
        "normal_HGS": n.tolist(),
        "normal_image_unit": image_n.tolist(),
        "ellipsoid_equation_residual": float(field(r, p)),
        "height_residual_Mm": float(np.linalg.norm(r) * R_MM - R_MM - height_mm),
    }
