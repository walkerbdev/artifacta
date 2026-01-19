"""Risk management utilities"""


def calculate_position_size(capital, risk_per_trade):
    return capital * risk_per_trade


def apply_stop_loss(position, stop_loss_pct):
    return position * (1 - stop_loss_pct)
