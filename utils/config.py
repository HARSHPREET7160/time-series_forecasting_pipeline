"""
Personalized setup derived from roll number (must be visible in code).

Roll number: 102317160

window_size = (sum of all digits) mod 10 + 8
prediction_horizon = (last 2 digits) mod 3 + 1
hidden_size = (first 3 digits) mod 16 + 8

I keep it here so sir can easily see my parameters are coming from roll number,
not I am randomly choosing.
"""

ROLL_NUMBER = "102317160"

_digits = [int(ch) for ch in ROLL_NUMBER if ch.isdigit()]

window_size = (sum(_digits) % 10) + 8
prediction_horizon = (int(ROLL_NUMBER[-2:]) % 3) + 1
hidden_size = (int(ROLL_NUMBER[:3]) % 16) + 8
