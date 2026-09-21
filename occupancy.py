"""
Find the pupil center by matching box occupancy grids between frames.

The Shack-Hartmann spots sit on the (fixed) lenslet lattice, so which lattice sites have a spot in a frame is the footprint of the
pupil on the lattice: about the same shape from frame to frame, but where the pupil is. Parts of it may be obstructed (on some
movies the bottom third), so some sites of the shape are simply empty.

  - A reference shape (the "template") is made from many frames of the movie: the reference frame is the one with the most spots,
    the other frames are matched to it, and the template is every site occupied in enough of the matched frames. So the
    template is the whole shape, not one frame's obstructed part of it.
  - Each frame's spots are put on the lattice sites, and the whole-site shift that best matches the frame's sites to the template
    is found: sites that agree are rewarded, spots outside the template are penalized hard (they don't belong), and template sites
    with no spot cost only a little (obstruction, or a dim spot).
  - The pupil center is the center site of the template moved by that shift, at the lattice position in this frame (which
    follows the spots if the lattice moves a little). The template's center is the middle of its own shape (halfway between its
    outermost sites, in each direction; a half-way point rounds up), so it doesn't depend on where anyone put a crosshair.

Frames with too few spots to say (a blink, a flash) get the template center unmoved.
"""
import numpy as np
from scipy import ndimage

PARAMS = dict(
    sigma=2.0,             # Smoothing (pixels) before looking for spots
    threshold=0.25,        # A spot is at least this fraction of the brightest (99.95th percentile) smoothed pixel
    max_shift=4,           # Most sites the pupil can have moved between the template and a frame
    outside_penalty=1.0,   # Cost of a spot outside the template's shape
    missing_penalty=0.25,  # Cost of a template site without a spot (obstruction)
    edge_reward=0.3,       # Reward for a spot on the template's edge: a partly hidden pupil fits inside the template at several
                           # positions, but only at the right one does its visible edge lie on the template's
    min_sites=12,          # Fewer occupied sites than this: can't tell, don't move the center
    sample_frames=100,     # Frames used to make the template
    template_fraction=0.25 # A template site is occupied in at least this fraction of the matched frames
)


def find_spots(image, pitch, params=PARAMS):
    """ (x, y) of each spot (local maximum of the smoothed image, at least a fraction of the brightest) as an (n, 2) array """
    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), params['sigma'])
    peak_level = np.percentile(smooth, 99.95)
    if not peak_level > 0:
        return np.zeros((0, 2))
    biggest = ndimage.maximum_filter(smooth, size=max(3, int(pitch * 0.6)))
    found = np.argwhere((smooth == biggest) & (smooth > params['threshold'] * peak_level))
    return found[:, ::-1].astype(float)


def lattice_phase(points, pitch):
    """ Where the lattice is: (x, y) of a lattice site, modulo the pitch, and how well the spots agree on it (0..1, 1 is perfect) """
    if len(points) == 0:
        return 0.0, 0.0, 0.0
    phases, coherence = [], []
    for d in (0, 1):
        z = np.mean(np.exp(2j * np.pi * points[:, d] / pitch))
        phases.append((np.angle(z) * pitch / (2 * np.pi)) % pitch)
        coherence.append(abs(z))
    return phases[0], phases[1], min(coherence)


def sites_of(points, origin, pitch):
    """ (set of the lattice sites that have a spot, (dx, dy)): sites are (i, j) whole numbers of pitches from `origin`.
        (dx, dy) is how far this frame's lattice is from the origin's (less than half a pitch): the spots are on lattice
        sites of the frame's own lattice, so that's taken out. """
    px, py, coherence = lattice_phase(points, pitch)
    if coherence < 0.3 or len(points) == 0:
        dx = dy = 0.0
    else:
        wrap = lambda d: (d + pitch / 2.0) % pitch - pitch / 2.0
        dx, dy = wrap(px - origin[0]), wrap(py - origin[1])
    sites = {(int(round((x - dx - origin[0]) / pitch)), int(round((y - dy - origin[1]) / pitch))) for x, y in points}
    return sites, (dx, dy)


def best_shift(sites, template, params=PARAMS):
    """ The (di, dj) that moves the template onto the frame's sites best, and what it scored (see the module docstring).
        Ties go to the smaller shift. Returns ((di, dj), info). """
    template = set(template)
    n_template = len(template)
    edge = [(i, j) for i, j in template if not ((i + 1, j) in template and (i - 1, j) in template and (i, j + 1) in template and (i, j - 1) in template)]
    m = int(params['max_shift'])
    best = None
    for di in range(-m, m + 1):
        for dj in range(-m, m + 1):
            inside = sum(1 for i, j in template if (i + di, j + dj) in sites)
            outside = len(sites) - inside
            missing = n_template - inside
            on_edge = sum(1 for i, j in edge if (i + di, j + dj) in sites)
            score = inside - params['outside_penalty'] * outside - params['missing_penalty'] * missing + params['edge_reward'] * on_edge
            key = (score, -(abs(di) + abs(dj)))
            if best is None or key > best[0]:
                best = (key, (di, dj), inside, outside, missing)
    (score, _), shift, inside, outside, missing = best
    return shift, dict(score=score, matched=inside, outside=outside, missing=missing, n_sites=len(sites), n_template=n_template)


def build_template(frames, pitch, params=PARAMS):
    """ frames: [(number, image)]. Returns the template (a dict), or None if no frame has enough spots. Its center_site is the
        middle of its shape: the template is the union of the frames' footprints, so an obstruction that hides part of the pupil
        in some frames doesn't move it. """
    observed = []
    for number, image in frames:
        points = find_spots(image, pitch, params)
        if len(points) >= params['min_sites']:
            observed.append((number, points))
    if not observed:
        return None
    ref_number, ref_points = max(observed, key=lambda o: len(o[1])) # The frame with the most spots (the first, if tied)
    px, py, _ = lattice_phase(ref_points, pitch)
    origin = (px, py)
    all_sites = [sites_of(points, origin, pitch)[0] for _, points in observed]
    template = set(sites_of(ref_points, origin, pitch)[0])

    for _ in range(2): # Match every frame to the template, and make a new one from what's occupied in enough of them
        counts = {}
        for sites in all_sites:
            (di, dj), _ = best_shift(sites, template, params)
            for i, j in sites:
                key = (i - di, j - dj) # In the template's coordinates
                counts[key] = counts.get(key, 0) + 1
        keep = {site for site, c in counts.items() if c >= params['template_fraction'] * len(all_sites)}
        if len(keep) >= params['min_sites']:
            template = keep

    # The template is in the coordinates of its reference frame. Its center: halfway between its outermost sites (a half-way
    # point, when the width is an even number of sites, rounds up)
    xs = [i for i, j in template]
    ys = [j for i, j in template]
    center_site = (int(np.floor((min(xs) + max(xs)) / 2.0 + 0.5)), int(np.floor((min(ys) + max(ys)) / 2.0 + 0.5)))
    return dict(sites=np.array(sorted(template), dtype=int), origin=origin, pitch=float(pitch), center_site=center_site,
                reference_frame=ref_number, n_frames=len(observed))


def template_center(template):
    """ (x, y) of the template's center, unmoved (for a frame that can't be judged) """
    return template['origin'][0] + template['center_site'][0] * template['pitch'], template['origin'][1] + template['center_site'][1] * template['pitch']


def find_center(image, template, params=PARAMS):
    """ (cx, cy, info): where the pupil center is in this frame, by matching its occupancy to the template """
    origin, pitch = template['origin'], template['pitch']
    points = find_spots(image, pitch, params)
    sites, (dx, dy) = sites_of(points, origin, pitch)
    if len(sites) < params['min_sites']:
        shift, info = (0, 0), dict(score=0.0, matched=0, outside=0, missing=0, n_sites=len(sites), n_template=len(template['sites']), ok=False)
    else:
        shift, info = best_shift(sites, [tuple(s) for s in template['sites']], params)
        info['ok'] = True
    ic, jc = template['center_site']
    info.update(shift=shift, phase=(dx, dy))
    return origin[0] + dx + (ic + shift[0]) * pitch, origin[1] + dy + (jc + shift[1]) * pitch, info
