"""Pure logic for the Flight Delay Risk Predictor: lookup, inbound what-if, scoring, labels."""
CATS = ["carrier", "origin_top", "dest_top", "origin_state", "dest_state", "inb_status_gate"]
NUMS = ["weekday", "dep_hour", "holiday_near", "crs_dep_min", "crs_arr_min", "crs_elapsed", "distance", "route_pad",
        "orig_dep_hour_n", "orig_dep_day_n", "dest_arr_hour_n", "carrier_day_n", "leg_num", "legs_day",
        "sched_turnaround", "te_carrier", "te_origin", "te_dest", "te_route", "te_carrier_origin", "te_origin_hour",
        "inb_known_gate"]
FEATURES = CATS + NUMS
