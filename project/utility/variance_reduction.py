import numpy as np

def standard_monte_carlo(simulated_forward_rates, strike, discount_factors):
    """
    Standard Monte Carlo pricing.
    :param simulated_forward_rates: Simulated forward rates from HJM model.
    :param strike: Strike rate for the cap or floor.
    :param discount_factors: Discount factors for cash flow present value.
    :return: Price and standard deviation.
    """
    payoffs = np.maximum(simulated_forward_rates - strike, 0)  # Cap payoff
    discounted_payoffs = np.sum(payoffs * discount_factors, axis=0)
    return np.mean(discounted_payoffs), np.std(discounted_payoffs)

def monte_carlo_with_antithetic(simulated_forward_rates, strike, discount_factors):
    """
    Monte Carlo pricing with Antithetic Variates.
    :param simulated_forward_rates: Simulated forward rates from HJM model.
    :param strike: Strike rate for the cap or floor.
    :param discount_factors: Discount factors for cash flow present value.
    :return: Price and standard deviation.
    """
    n_steps, n_maturities, n_simulations = simulated_forward_rates.shape
    payoffs = []

    for i in range(n_simulations):
        # Original path
        fwd_rate = simulated_forward_rates[:, :, i]
        payoff = np.maximum(fwd_rate - strike, 0)
        discounted_payoff = np.sum(payoff * discount_factors)
        
        # Antithetic path
        fwd_rate_antithetic = 2 * np.mean(simulated_forward_rates, axis=2) - fwd_rate
        payoff_antithetic = np.maximum(fwd_rate_antithetic - strike, 0)
        discounted_payoff_antithetic = np.sum(payoff_antithetic * discount_factors)
        
        # Average the payoffs
        payoffs.append(0.5 * (discounted_payoff + discounted_payoff_antithetic))
    
    return np.mean(payoffs), np.std(payoffs)

def monte_carlo_with_control_variates(simulated_forward_rates, strike, discount_factors, analytical_price):
    """
    Monte Carlo pricing with Control Variates.
    :param simulated_forward_rates: Simulated forward rates from HJM model.
    :param strike: Strike rate for the cap or floor.
    :param discount_factors: Discount factors for cash flow present value.
    :param analytical_price: Known analytical price (control variate).
    :return: Price and standard deviation.
    """
    payoffs = np.maximum(simulated_forward_rates - strike, 0)  # Cap payoff
    discounted_payoffs = np.sum(payoffs * discount_factors, axis=0)
    
    mc_price = np.mean(discounted_payoffs)
    control_correction = analytical_price - np.mean(analytical_price)  # Difference between actual and simulated
    adjusted_price = mc_price + control_correction
    return adjusted_price, np.std(discounted_payoffs)

def price_with_variance_reduction(method, simulated_forward_rates, strike, discount_factors, analytical_price=None):
    """
    Dispatch function to select variance reduction method.
    :param method: Variance reduction method ("standard", "antithetic", "control").
    :param simulated_forward_rates: Simulated forward rates from HJM model.
    :param strike: Strike rate for the cap or floor.
    :param discount_factors: Discount factors for cash flow present value.
    :param analytical_price: Known analytical price (for control variates).
    :return: Price and standard deviation.
    """
    if method == "standard":
        return standard_monte_carlo(simulated_forward_rates, strike, discount_factors)
    elif method == "antithetic":
        return monte_carlo_with_antithetic(simulated_forward_rates, strike, discount_factors)
    elif method == "control":
        if analytical_price is None:
            raise ValueError("Analytical price is required for control variates.")
        return monte_carlo_with_control_variates(simulated_forward_rates, strike, discount_factors, analytical_price)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'standard', 'antithetic', 'control'.")