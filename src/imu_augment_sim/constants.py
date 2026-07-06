"""Project wide constants."""

INTRINSIC_EULER_ORDER = 'XYZ'

# scipy uses scalar-last (x, y, z, w) quaternion ordering.
QUATERNION_AXES = ['i', 'j', 'k', 'w']

# OpenSim expects scalar-first (w, x, y, z) quaternion ordering in its .sto files.
OPENSIM_QUATERNION_AXES = ['w', 'i', 'j', 'k']

# Threshold (in degrees) above which a step-to-step change in an Euler angle
# series is treated as a wrap-around discontinuity rather than real motion.
EULER_JUMP_THRESHOLD_DEG = 330

# Euclidean distance between consecutive (unit) quaternions above which a
# sign-flip glitch is assumed.
GLITCH_DISTANCE_THRESHOLD = 1.5

# Full turn / half turn in degrees, used when wrapping Euler angles.
FULL_TURN_DEG = 360
HALF_TURN_DEG = 180
