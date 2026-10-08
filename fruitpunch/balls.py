"""Projectile catalog: size, mass, look, and how each one is thrown.

dist: throw distance from the robot (m); release: launch height (m); flight: time to reach
the aim point (s); gscale: gravity on the projectile as a fraction of g (the straight tomato
is a game-style floater); sticky: welds to the torso/head on contact.
"""

BALLS = {
    "tennis":     dict(radius=0.033, mass=0.058, rgba=(0.85, 0.95, 0.2, 1), dist=(3.0, 5.0),
                       release=(1.0, 1.6), flight=(0.5, 0.8), gscale=1.0, sticky=False),
    "dodgeball":  dict(radius=0.105, mass=0.30, rgba=(0.75, 0.15, 0.55, 1), dist=(4.0, 6.0),
                       release=(1.0, 1.6), flight=(0.6, 0.9), gscale=1.0, sticky=False),
    "volleyball": dict(radius=0.105, mass=0.27, rgba=(0.95, 0.95, 0.9, 1), dist=(6.0, 9.0),
                       release=(1.8, 2.4), flight=(1.1, 1.5), gscale=1.0, sticky=False),
    "basketball": dict(radius=0.12, mass=0.62, rgba=(0.9, 0.42, 0.1, 1), dist=(6.0, 9.0),
                       release=(1.8, 2.4), flight=(1.1, 1.5), gscale=1.0, sticky=False),
    "cannonball": dict(radius=0.08, mass=5.0, rgba=(0.12, 0.12, 0.13, 1), dist=(25.0, 40.0),
                       release=(0.5, 1.0), flight=(2.6, 3.4), gscale=1.0, sticky=False),
    "tomato":     dict(radius=0.035, mass=0.12, rgba=(0.85, 0.08, 0.05, 1), dist=(8.0, 10.0),
                       release=(0.8, 1.2), flight=(1.6, 2.0), gscale=0.08, sticky=True),
    "tomato_lob": dict(radius=0.035, mass=0.12, rgba=(0.85, 0.08, 0.05, 1), dist=(6.0, 8.0),
                       release=(1.2, 1.8), flight=(1.3, 1.6), gscale=1.0, sticky=True),
}
