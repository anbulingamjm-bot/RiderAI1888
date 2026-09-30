"""Healthy TVS Raider 125 operating relationships (used for data generation and simulation)."""
def rpm_f(speed): return 1200 + 55 * speed
def temp_f(rpm, speed): return 50 + 0.005 * rpm + 0.08 * speed
def batt_f(rpm): return 12.8 + 0.0003 * rpm
def fuel_f(rpm, speed): return 0.3 + 0.0004 * rpm + 0.005 * speed
