# simulation/train_movement.py
import math

def compute_train_positions(trains_df, current_time):
    positions = []
    for _, r in trains_df.iterrows():
        if current_time < r.start_time:
            x, y = r.x_source, r.y_source
        elif current_time > r.arrive_time:
            x, y = r.x_platform, r.y_platform
        else:
            frac = (current_time - r.start_time) / (r.arrive_time - r.start_time)
            x = r.x_source + frac * (r.x_platform - r.x_source)
            y = r.y_source + frac * (r.y_platform - r.y_source)
        positions.append((r.train_id, x, y))
    return positions
