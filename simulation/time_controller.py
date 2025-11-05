# simulation/time_controller.py
# Minimal time utilities; expand later for playback/animation
def clamp_time(t, min_t=0, max_t=24*60):
    return max(min_t, min(max_t, t))
