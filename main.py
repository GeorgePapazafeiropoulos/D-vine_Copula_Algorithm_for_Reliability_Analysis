"""
Simplified D-vine Copula Implementation and Verification
Tests GAUSSIAN, CLAYTON, GUMBEL, FRANK, and INDEPENDENCE copula families
Simple limit state function of 3 variables is used: G = threshold - (X1 + X2^2 + X3)
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional

# -----------------------------
# D-vine Copula Implementation
# -----------------------------

class CopulaFamily(Enum):
    """Available pair-copula families for D-vine construction."""
    GAUSSIAN = "gaussian"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    INDEPENDENCE = "independence"

class PairCopula:
    """Represents a pair-copula with parameter estimation and selection."""

    def __init__(self, family: CopulaFamily, theta: float = None):
        self.family = family
        self.theta = theta
        self.log_likelihood = 0.0
        self.aic = float('inf')
        self.bic = float('inf')

    def fit(self, u1: np.ndarray, u2: np.ndarray) -> None:
        """Fit copula parameters using Maximum Likelihood Estimation (MLE)."""
        if self.family == CopulaFamily.INDEPENDENCE:
            self.theta = 0.0
            self._calculate_fit_metrics(u1, u2)
            return

        # Initial parameter guess based on Kendall's tau
        tau_result = stats.kendalltau(u1, u2)
        tau = tau_result[0]
        if np.isnan(tau) or tau is None:
            tau = 0.0
        tau = np.clip(tau, -0.999, 0.999) # Clip tau to avoid extreme values
        initial_guess = self._tau_to_theta(tau)

        # Define log-likelihood function for each copula family
        def neg_log_likelihood(theta):
            return -self._log_likelihood(theta, u1, u2)

        # Bounds for parameters
        bounds = self._get_parameter_bounds()

        try:
            # Try bounded optimization
            if bounds[0] < bounds[1]:
                result = optimize.minimize_scalar(
                    neg_log_likelihood,
                    bounds=bounds,
                    method='bounded',
                    options={'maxiter': 1000}
                )
                self.theta = result.x
            else:
                # If bounds are invalid, use initial guess
                self.theta = initial_guess

            self._calculate_fit_metrics(u1, u2)
        except Exception as e:
            # Fall back to independence if fitting fails
            self.family = CopulaFamily.INDEPENDENCE
            self.theta = 0.0
            self._calculate_fit_metrics(u1, u2)

    def _log_likelihood(self, theta: float, u1: np.ndarray, u2: np.ndarray) -> float:
        """Calculate log-likelihood for given parameters."""
        n = len(u1)

        if self.family == CopulaFamily.GAUSSIAN:
            # Gaussian copula density
            if abs(theta) >= 1:
                return -np.inf

            x1 = stats.norm.ppf(np.clip(u1, 1e-10, 1-1e-10))
            x2 = stats.norm.ppf(np.clip(u2, 1e-10, 1-1e-10))

            rho = theta  # For Gaussian copula, theta IS the correlation
            if abs(rho) >= 1:
                return -np.inf

            # Log-likelihood for bivariate normal
            ll = -0.5 * np.log(1 - rho**2) - 0.5/(1 - rho**2) * (
                x1**2 + x2**2 - 2 * rho * x1 * x2
            ) + 0.5 * (x1**2 + x2**2)

            return np.sum(ll)

        elif self.family == CopulaFamily.CLAYTON:
            # Clayton copula density
            if theta <= 0:
                return -np.inf

            # Avoid numerical issues
            u1_clip = np.clip(u1, 1e-10, 1-1e-10)
            u2_clip = np.clip(u2, 1e-10, 1-1e-10)

            term = u1_clip**(-theta) + u2_clip**(-theta) - 1
            if np.any(term <= 0):
                return -np.inf

            log_c = np.log(1 + theta) - (1 + theta) * (np.log(u1_clip) + np.log(u2_clip)) - (1/theta + 2) * np.log(term)
            return np.sum(log_c)

        elif self.family == CopulaFamily.GUMBEL:
            # Gumbel copula density
            if theta < 1:
                return -np.inf

            # Avoid numerical issues
            u1_clip = np.clip(u1, 1e-10, 1-1e-10)
            u2_clip = np.clip(u2, 1e-10, 1-1e-10)

            log_u1 = -np.log(u1_clip)
            log_u2 = -np.log(u2_clip)

            # Use logarithms to avoid overflow for large theta
            t = log_u1**theta + log_u2**theta
            t = np.maximum(t, 1e-10)

            # C(u,v) = exp(-t^(1/theta))
            A = t**(1/theta)
            log_C = -A

            # Log-density: log(c) = log(C) + log((log_u1*log_u2)^(theta-1)) +
            #                    log(t^(1/theta - 2)) + log(1 + (theta-1)*t^(-1/theta))
            log_density = (log_C +
                          (theta - 1) * (np.log(log_u1) + np.log(log_u2)) +
                          (1/theta - 2) * np.log(t) +
                          np.log(1 + (theta - 1) * t**(-1/theta)))

            return np.sum(log_density)

        elif self.family == CopulaFamily.FRANK:
            # Frank copula density
            if abs(theta) < 1e-10:
                return 0.0  # Independence

            # Avoid numerical issues
            u1_clip = np.clip(u1, 1e-10, 1-1e-10)
            u2_clip = np.clip(u2, 1e-10, 1-1e-10)

            # For small theta, use approximation to avoid overflow
            if abs(theta) < 0.1:
                # Independence approximation for small theta
                return 0.0

            exp_neg_theta = np.exp(-theta)
            exp_neg_theta_u1 = np.exp(-theta * u1_clip)
            exp_neg_theta_u2 = np.exp(-theta * u2_clip)

            denominator = exp_neg_theta_u1 * exp_neg_theta_u2 - exp_neg_theta_u1 - exp_neg_theta_u2 + exp_neg_theta

            # Avoid division by zero
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)

            # Frank copula density: c(u,v) = θ(1-exp(-θ))exp(-θ(u+v)) / [exp(-θ) - (1-exp(-θu))(1-exp(-θv))]^2
            numerator = theta * (1 - exp_neg_theta) * exp_neg_theta_u1 * exp_neg_theta_u2

            log_c = np.log(np.abs(numerator)) - 2 * np.log(np.abs(denominator))
            return np.sum(log_c)

        return 0.0

    def _calculate_fit_metrics(self, u1: np.ndarray, u2: np.ndarray) -> None:
        """Calculate AIC and BIC for model selection."""
        n = len(u1)
        ll = self._log_likelihood(self.theta, u1, u2)
        self.log_likelihood = ll

        # Number of parameters
        k = 1 if self.family != CopulaFamily.INDEPENDENCE else 0

        # AIC and BIC
        self.aic = 2 * k - 2 * ll
        self.bic = k * np.log(n) - 2 * ll

    def _tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to initial parameter guess."""
        bounds = self._get_parameter_bounds()
        lower_bound, upper_bound = bounds

        # Clip tau to avoid extreme values
        tau = np.clip(tau, -0.999, 0.999)

        if self.family == CopulaFamily.GAUSSIAN:
            # For Gaussian: τ = (2/π)arcsin(ρ) => ρ = sin(πτ/2)
            # Transform Kendall's tau to correlation
            if abs(tau) > 0.999:
                return np.sign(tau) * 0.999
            return np.sin(tau * np.pi / 2)

        elif self.family == CopulaFamily.CLAYTON:
            # For Clayton: τ = θ/(θ+2) => θ = 2τ/(1-τ)
            if tau >= 0.999:
                return upper_bound * 0.9
            elif tau > 0:
                theta = 2 * tau / (1 - tau)
                return np.clip(theta, lower_bound, upper_bound)
            else:
                return max(lower_bound, 0.001)

        elif self.family == CopulaFamily.GUMBEL:
            # For Gumbel: τ = 1 - 1/θ => θ = 1/(1-τ)
            if tau >= 0.999:
                return upper_bound * 0.9
            elif tau > 0:
                theta = 1 / (1 - tau)
                return np.clip(theta, lower_bound, upper_bound)
            else:
                return max(lower_bound, 1.001)

        elif self.family == CopulaFamily.FRANK:
            # For Frank: τ = 1 + 4[D₁(θ)-1]/θ where D₁ is Debye function
            # Simple approximation for initialization
            if abs(tau) < 0.33:
                return 3 * tau
            elif tau > 0.999:
                return upper_bound * 0.9
            elif tau < -0.999:
                return lower_bound * 0.9
            else:
                return 9 * tau

        return 0.0

    def _get_parameter_bounds(self) -> Tuple[float, float]:
        """Get parameter bounds for optimization."""
        if self.family == CopulaFamily.GAUSSIAN:
            return (-0.99, 0.99)
        elif self.family == CopulaFamily.CLAYTON:
            return (0.001, 20.0)
        elif self.family == CopulaFamily.GUMBEL:
            return (1.001, 20.0)
        elif self.family == CopulaFamily.FRANK:
            return (-30.0, 30.0)  # Reduced from ±50 for better stability
        return (0.0, 0.0)

    def h_function(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """h-function for conditional distribution C_{2|1}(u2|u1).
        Note: 
        - u1 is the CONDITIONING variable (given/known)
        - u2 is the CONDITIONED variable (what we're computing distribution for)
        - Returns P(U2 ≤ u2 | U1 = u1) = ∂C(u1, u2)/∂u1
        """
        if self.family == CopulaFamily.INDEPENDENCE:
            return u2

        u1_clip = np.clip(u1, 1e-10, 1-1e-10)
        u2_clip = np.clip(u2, 1e-10, 1-1e-10)

        if self.family == CopulaFamily.GAUSSIAN:
            # Gaussian h-function
            x1 = stats.norm.ppf(u1_clip)
            x2 = stats.norm.ppf(u2_clip)
            rho = self.theta  # For Gaussian, theta is the correlation
            return stats.norm.cdf((x2 - rho * x1) / np.sqrt(1 - rho**2))

        elif self.family == CopulaFamily.CLAYTON:
            # Clayton h-function
            if self.theta < 1e-10:
                return u2_clip

            term = u1_clip**(-self.theta) + u2_clip**(-self.theta) - 1
            valid = term > 0

            result = np.zeros_like(u1_clip)
            result[valid] = u1_clip[valid]**(-self.theta - 1) * term[valid]**(-1/self.theta - 1)
            result[~valid] = np.where(u2_clip[~valid] >= 1-1e-10, 1.0, 0.0)

            return np.clip(result, 0.0, 1.0)

        elif self.family == CopulaFamily.GUMBEL:
            # Gumbel h-function
            if self.theta < 1.001:
                return u2_clip

            log_u1 = -np.log(u1_clip)
            log_u2 = -np.log(u2_clip)

            t = log_u1**self.theta + log_u2**self.theta
            t = np.maximum(t, 1e-10)

            A = t**(1/self.theta)
            C = np.exp(-A)

            # Correct formula: h(v|u) = C(u,v)/u * (log_u)^(θ-1) / t^((θ-1)/θ)
            h = C / u1_clip * (log_u1**(self.theta - 1) / t**((self.theta - 1)/self.theta))
            return np.clip(h, 0.0, 1.0)

        elif self.family == CopulaFamily.FRANK:
            # Frank h-function
            if abs(self.theta) < 1e-10:
                return u2_clip

            exp_neg_theta = np.exp(-self.theta)
            exp_neg_theta_u1 = np.exp(-self.theta * u1_clip)
            exp_neg_theta_u2 = np.exp(-self.theta * u2_clip)

            numerator = (exp_neg_theta_u2 - 1) * exp_neg_theta_u1
            denominator = (exp_neg_theta_u1 - 1) * (exp_neg_theta_u2 - 1) + (exp_neg_theta - 1)

            # Handle potential division by zero
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)

            h = numerator / denominator
            return np.clip(h, 0.0, 1.0)

        return u2_clip

    def inverse_h_function(self, u1: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse h-function for conditional sampling."""
        if self.family == CopulaFamily.INDEPENDENCE:
            return v

        u1_clip = np.clip(u1, 1e-10, 1-1e-10)
        v_clip = np.clip(v, 1e-10, 1-1e-10)

        if self.family == CopulaFamily.GAUSSIAN:
            # Gaussian inverse
            x1 = stats.norm.ppf(u1_clip)
            rho = self.theta
            z = stats.norm.ppf(v_clip)
            x2 = rho * x1 + np.sqrt(1 - rho**2) * z
            return stats.norm.cdf(x2)

        elif self.family == CopulaFamily.CLAYTON:
            # Clayton inverse
            if self.theta < 1e-10:
                return v_clip

            # v = [(p * u^{θ+1})^{-θ/(θ+1)} - u^{-θ} + 1]^{-1/θ}
            term1 = (v_clip * u1_clip**(self.theta + 1))**(-self.theta/(self.theta + 1))
            term2 = u1_clip**(-self.theta)
            v_pow_neg_theta = term1 - term2 + 1

            v_pow_neg_theta = np.maximum(v_pow_neg_theta, 1e-10)
            u2 = v_pow_neg_theta**(-1/self.theta)
            return np.clip(u2, 0.0, 1.0)

        elif self.family == CopulaFamily.GUMBEL:
            # Gumbel inverse - numerical solution
            if self.theta < 1.001:
                return v_clip

            # Define root function
            def root_func(u2_val, u1_val, v_val):
                h_val = self.h_function(u1_val, u2_val)
                return h_val - v_val

            # Use vectorized root finding where possible
            result = np.zeros_like(u1_clip)
            for i in range(len(u1_clip)):
                try:
                    root = optimize.brentq(
                        lambda x: root_func(x, u1_clip[i], v_clip[i]),
                        1e-10, 1-1e-10,
                        maxiter=100,
                        xtol=1e-8
                    )
                    result[i] = root
                except:
                    result[i] = v_clip[i]  # Fallback

            return np.clip(result, 0.0, 1.0)

        elif self.family == CopulaFamily.FRANK:
            # Frank inverse - numerical solution
            if abs(self.theta) < 1e-10:
                return v_clip
            
            # Define root function
            def root_func(u2_val, u1_val, v_val):
                h_val = self.h_function(u1_val, u2_val)
                return h_val - v_val
            
            # Use vectorized root finding where possible
            result = np.zeros_like(u1_clip)
            for i in range(len(u1_clip)):
                try:
                    root = optimize.brentq(
                        lambda x: root_func(x, u1_clip[i], v_clip[i]),
                        1e-10, 1-1e-10,
                        maxiter=100,
                        xtol=1e-8
                    )
                    result[i] = root
                except:
                    result[i] = v_clip[i]  # Fallback
            
            return np.clip(result, 0.0, 1.0)

        return v_clip

class DVineStructure:
    """Complete D-vine structure with hierarchical construction."""

    def __init__(self, order: List[str]):
        self.order = order
        self.n_vars = len(order)
        self.trees: List[Dict[Tuple[int, int], PairCopula]] = []
        self.vine_array: Optional[np.ndarray] = None

    def build_from_data(self, data: np.ndarray, families: List[CopulaFamily] = None) -> None:
        """
        Build D-vine structure from data using hierarchical iteration with AIC/BIC selection.

        Parameters:
        -----------
        data : np.ndarray
            Array of shape (n_samples, n_vars) with variables in the order of self.order
        families : List[CopulaFamily]
            List of copula families to consider. If None, uses all families.
        """
        if families is None:
            families = [CopulaFamily.GAUSSIAN, CopulaFamily.CLAYTON,
                       CopulaFamily.GUMBEL, CopulaFamily.FRANK]

        n_samples, n_vars = data.shape

        # Transform to uniform marginals using empirical CDF
        u_data = np.zeros_like(data)
        for i in range(n_vars):
            u_data[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)

        # Initialize vine array for storing pair-copulas
        self.vine_array = np.full((n_vars, n_vars), None, dtype=object)

        print("\nBuilding D-vine structure...")

        # Tree 1: unconditional pairs
        tree1 = {}
        print("\nTree 1 (unconditional pairs):")

        for i in range(n_vars - 1):
            u1 = u_data[:, i]
            u2 = u_data[:, i + 1]

            # Try all copula families and select best based on BIC
            best_copula = None
            best_bic = float('inf')

            for family in families:
                copula = PairCopula(family)
                copula.fit(u1, u2)

                if copula.bic < best_bic:
                    best_bic = copula.bic
                    best_copula = copula

            # Check if the best copula is close to independence
            if (best_copula.family == CopulaFamily.GAUSSIAN and abs(best_copula.theta) < 0.01) or \
            (best_copula.family == CopulaFamily.CLAYTON and best_copula.theta < 0.01) or \
            (best_copula.family == CopulaFamily.GUMBEL and best_copula.theta < 1.01) or \
            (best_copula.family == CopulaFamily.FRANK and abs(best_copula.theta) < 0.01):
                
                best_copula = PairCopula(CopulaFamily.INDEPENDENCE)
                best_copula.theta = 0.0
                best_copula._calculate_fit_metrics(u1, u2)

            tree1[(i, i + 1)] = best_copula
            self.vine_array[i, i + 1] = best_copula

            print(f"  Pair ({self.order[i]}, {self.order[i+1]}): {best_copula.family.value}, "
                  f"θ={best_copula.theta:.3f}, BIC={best_copula.bic:.1f}")

        self.trees.append(tree1)

        # Higher trees: conditional pairs
        current_data = u_data.copy()

        for tree_level in range(2, n_vars):
            tree = {}
            print(f"\nTree {tree_level} (conditional pairs):")

            for i in range(n_vars - tree_level):
                j = i + tree_level

                # Apply h-function recursively through previous trees
                u_cond1 = current_data[:, i]
                u_cond2 = current_data[:, j]

                # Apply conditioning through intermediate variables
                for k in range(tree_level - 1):
                    intermediate = i + k + 1
                    # Get copula from previous tree
                    copula1 = self.vine_array[i, intermediate]
                    copula2 = self.vine_array[intermediate, j]

                    if copula1 is not None:
                        u_cond1 = copula1.h_function(current_data[:, intermediate], u_cond1)
                    if copula2 is not None:
                        u_cond2 = copula2.h_function(current_data[:, intermediate], u_cond2)

                # Try all copula families
                best_copula = None
                best_bic = float('inf')

                for family in families:
                    copula = PairCopula(family)
                    copula.fit(u_cond1, u_cond2)

                    if copula.bic < best_bic:
                        best_bic = copula.bic
                        best_copula = copula

                # Check if the best copula is close to independence
                if best_copula.family == CopulaFamily.GAUSSIAN and abs(best_copula.theta) < 0.01:
                    best_copula = PairCopula(CopulaFamily.INDEPENDENCE)
                    best_copula.theta = 0.0
                    best_copula._calculate_fit_metrics(u_cond1, u_cond2)
                elif best_copula.family == CopulaFamily.CLAYTON and best_copula.theta < 0.01:
                    best_copula = PairCopula(CopulaFamily.INDEPENDENCE)
                    best_copula.theta = 0.0
                    best_copula._calculate_fit_metrics(u_cond1, u_cond2)
                elif best_copula.family == CopulaFamily.GUMBEL and best_copula.theta < 1.01:
                    best_copula = PairCopula(CopulaFamily.INDEPENDENCE)
                    best_copula.theta = 0.0
                    best_copula._calculate_fit_metrics(u_cond1, u_cond2)
                elif best_copula.family == CopulaFamily.FRANK and abs(best_copula.theta) < 0.01:
                    best_copula = PairCopula(CopulaFamily.INDEPENDENCE)
                    best_copula.theta = 0.0
                    best_copula._calculate_fit_metrics(u_cond1, u_cond2)

                tree[(i, j)] = best_copula
                self.vine_array[i, j] = best_copula

                print(f"  Pair ({self.order[i]}, {self.order[j]}): {best_copula.family.value}, "
                      f"θ={best_copula.theta:.3f}, BIC={best_copula.bic:.1f}")

            self.trees.append(tree)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the D-vine structure using recursive algorithm."""
        if self.vine_array is None:
            raise ValueError("D-vine must be built before sampling")

        n_vars = self.n_vars
        samples = np.zeros((n, n_vars))

        # Sample first variable
        samples[:, 0] = rng.uniform(size=n)

        # Sample subsequent variables using conditional distributions
        for j in range(1, n_vars):
            # Generate random probability V ~ Uniform(0,1)
            v = rng.uniform(size=n)

            # Apply inverse h-functions with proper conditioning
            for i in range(j):  # From 0 to j-1
                # Get the conditioned version of U_i (u_cond) given i+1 to j-1
                u_cond = samples[:, i]

                # Apply h-functions to condition U_i on the intermediate variables
                for k in range(i, j-1):
                    copula_cond = self.vine_array[k, k+1]
                    if copula_cond is not None:
                        # h(u1, u2) with u1=conditioning, u2=conditioned
                        u_cond = copula_cond.h_function(samples[:, k+1], u_cond)  

                # Now u_cond is U_i | i+1,...,j-1
                # Apply inverse h-function of the pair (i,j)
                copula = self.vine_array[i, j]
                if copula is not None:
                    v = copula.inverse_h_function(u_cond, v)

            samples[:, j] = v

        return samples

# -----------------------------
# Reliability Analysis
# -----------------------------

@dataclass
class RandomVariable:
    """Simple random variable for reliability analysis."""
    name: str
    mean: float
    std: float
    
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample n values (assuming normal distribution)."""
        return rng.normal(self.mean, self.std, n)

def create_reliability_variables() -> List[RandomVariable]:
    """Create three independent random variables for reliability analysis."""
    return [
        RandomVariable("X1", mean=2.0, std=0.4),
        RandomVariable("X2", mean=3.0, std=0.6), 
        RandomVariable("X3", mean=1.5, std=0.3)
    ]

def create_correlated_samples(n_samples: int, seed: int = 42) -> np.ndarray:
    """Create correlated samples with known copula structures for testing.
    Returns samples with:
    - X1: Normal RV
    - X2 | X1: Gaussian copula (rho=0.5)
    - X3 | X2: Clayton copula (theta=2.0)
    """
    rng = np.random.default_rng(seed)
    
    # Create base random variables
    variables = create_reliability_variables()
    
    # Sample from a vine structure with known copulas
    # Create samples with known correlations using the copula classes directly
    
    # Generate independent uniforms
    u_indep = rng.uniform(size=(n_samples, 3))
    
    # Apply copula transformations
    samples = np.zeros((n_samples, 3))
    
    # X1: normal
    samples[:, 0] = variables[0].mean + variables[0].std * stats.norm.ppf(u_indep[:, 0])
    
    # X2 | X1: Gaussian
    copula_gauss = PairCopula(CopulaFamily.GAUSSIAN, theta=0.5)
    u2_cond = copula_gauss.inverse_h_function(u_indep[:, 0], u_indep[:, 1])
    samples[:, 1] = variables[1].mean + variables[1].std * stats.norm.ppf(u2_cond)
    
    # X3 | X2: Clayton  
    copula_clayton = PairCopula(CopulaFamily.CLAYTON, theta=2.0)
    u3_cond = copula_clayton.inverse_h_function(u_indep[:, 1], u_indep[:, 2])
    samples[:, 2] = variables[2].mean + variables[2].std * stats.norm.ppf(u3_cond)
    
    return samples

def limit_state_function(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """Limit state function: G = threshold - (X1 + X2^2 + X3)
    
    Failure when G <= 0, i.e., X1 + X2^2 + X3 >= threshold
    """
    return threshold - (x1 + x2**2 + x3)

def reliability_analysis_independent(n_samples: int = 10000, threshold: float = 20.0):
    """Perform reliability analysis using independent sampling."""
    print("\nRELIABILITY ANALYSIS USING INDEPENDENT SAMPLING")
    
    # Create random variables
    variables = create_reliability_variables()
    
    print(f"\nRandom Variables:")
    for var in variables:
        print(f"  {var.name}: N({var.mean:.1f}, {var.std:.1f}²)")
    
    print(f"\nLimit state: G = {threshold:.1f} - f(X1 , X2 , X3)")
    print(f"Failure when: f(X1 , X2 , X3) >= {threshold:.1f}")
    
    # Sample independently
    rng = np.random.default_rng(42)
    print(f"\nSampling {n_samples} points independently...")
    samples = np.zeros((n_samples, len(variables)))
    for i, var in enumerate(variables):
        samples[:, i] = var.sample(rng, n_samples)
    
    # Evaluate limit state
    G = limit_state_function(samples[:, 0], samples[:, 1], samples[:, 2], threshold)
    failures = G <= 0
    pf = np.mean(failures)
    
    # Calculate confidence interval
    std_pf = np.sqrt(pf * (1 - pf) / n_samples)
    # CI: [mean-z_0.05*std,mean+z_0.05*std]
    # z_a is the (1-a/2) quantile of a standard normal distribution
    # z_0.05 = 1.96
    ci_lower = max(0, pf - 1.96 * std_pf)
    ci_upper = min(1, pf + 1.96 * std_pf)
    
    print(f"\nResults:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Failures: {np.sum(failures)}")
    print(f"  Failure probability: {pf:.6f}")
    print(f"  95% Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  CoV of estimate: {std_pf/pf:.4f}")
        
    return pf

def reliability_analysis_independent_dvine(n_samples: int = 10000, n_train: int = 5000, threshold: float = 20.0):
    """Perform reliability analysis using D-vine copula. It selects appropriate copulas among GAUSSIAN, CLAYTON, GUMBEL and FRANK"""
    print("\nRELIABILITY ANALYSIS USING D-VINE COPULA ON INDEPENDENT DATA")
    
    # Create random variables
    variables = create_reliability_variables()
    var_names = [v.name for v in variables]
    
    print(f"\nRandom Variables:")
    for var in variables:
        print(f"  {var.name}: N({var.mean:.1f}, {var.std:.1f}²)")
    
    print(f"\nLimit state: G = {threshold:.1f} - f(X1 , X2 , X3)")
    print(f"Failure when: f(X1 , X2 , X3) >= {threshold:.1f}")
    
    # Generate training data for D-vine fitting
    rng = np.random.default_rng(42)
    
    print(f"\nGenerating {n_train} training samples...")
    train_data = np.zeros((n_train, len(variables)))
    for i, var in enumerate(variables):
        train_data[:, i] = var.sample(rng, n_train)
    
    # Build D-vine (will select appropriate copulas among GAUSSIAN, CLAYTON, GUMBEL and FRANK)
    dvine = DVineStructure(var_names)
    dvine.build_from_data(train_data)
    
    # Sample from D-vine
    print(f"\nSampling {n_samples} points from D-vine...")
    u_samples = dvine.sample(n_samples, rng)
    
    # Transform back to physical space
    samples = np.zeros((n_samples, len(variables)))
    for i, var in enumerate(variables):
        # Transform uniform to normal using inverse CDF
        z_samples = stats.norm.ppf(np.clip(u_samples[:, i], 1e-10, 1-1e-10))
        samples[:, i] = var.mean + var.std * z_samples
    
    # Evaluate limit state
    G = limit_state_function(samples[:, 0], samples[:, 1], samples[:, 2], threshold)
    failures = G <= 0
    pf = np.mean(failures)
    
    # Calculate confidence interval
    std_pf = np.sqrt(pf * (1 - pf) / n_samples)
    ci_lower = max(0, pf - 1.96 * std_pf)
    ci_upper = min(1, pf + 1.96 * std_pf)
    
    print(f"\nResults:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Failures: {np.sum(failures)}")
    print(f"  Failure probability: {pf:.6f}")
    print(f"  95% Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  CoV of estimate: {std_pf/pf:.4f}")
        
    return pf

def reliability_analysis_correlated_dvine(n_samples: int = 10000, n_train: int = 5000, threshold: float = 20.0):
    """Perform reliability analysis using D-vine copula on correlated data."""
    print("\nRELIABILITY ANALYSIS USING D-VINE COPULA ON CORRELATED DATA")
    
    # Create random variables (for reference)
    variables = create_reliability_variables()
    var_names = [v.name for v in variables]
    
    print(f"\nRandom Variables (marginals):")
    for var in variables:
        print(f"  {var.name}: N({var.mean:.1f}, {var.std:.1f}²)")
    
    print(f"\nKnown correlations:")
    print(f"  X1: Normal RV")
    print(f"  X2|X1: Gaussian copula (rho=0.5)")
    print(f"  X3|X2: Clayton copula (theta=2.0)")
    
    print(f"\nLimit state: G = {threshold:.1f} - f(X1 , X2 , X3)")
    print(f"Failure when: f(X1 , X2 , X3) >= {threshold:.1f}")
    
    # Generate training data with known correlations
    rng = np.random.default_rng(42)
    
    print(f"\nGenerating {n_train} training samples with known correlations...")
    train_data = create_correlated_samples(n_train, seed=42)
    
    # Build D-vine (should identify the underlying copula structure)
    dvine = DVineStructure(var_names)
    dvine.build_from_data(train_data)
    
    # Sample from D-vine
    print(f"\nSampling {n_samples} points from fitted D-vine...")
    u_samples = dvine.sample(n_samples, rng)
    
    # Transform back to physical space
    samples = np.zeros((n_samples, len(variables)))
    for i, var in enumerate(variables):
        # Transform uniform to normal using inverse CDF
        z_samples = stats.norm.ppf(np.clip(u_samples[:, i], 1e-10, 1-1e-10))
        samples[:, i] = var.mean + var.std * z_samples
    
    # Evaluate limit state
    G = limit_state_function(samples[:, 0], samples[:, 1], samples[:, 2], threshold)
    failures = G <= 0
    pf = np.mean(failures)
    
    # Calculate confidence interval
    std_pf = np.sqrt(pf * (1 - pf) / n_samples)
    ci_lower = max(0, pf - 1.96 * std_pf)
    ci_upper = min(1, pf + 1.96 * std_pf)
    
    print(f"\nResults:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Failures: {np.sum(failures)}")
    print(f"  Failure probability: {pf:.6f}")
    print(f"  95% Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  CoV of estimate: {std_pf/pf:.4f}")
        
    return pf


# -----------------------------
# Main
# -----------------------------

def main():

    # Perform reliability analysis
    print("\n" + "="*60)
    print("RELIABILITY ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Test 1: Independent variables
    print("\n" + "="*60)
    print("TEST 1: INDEPENDENT VARIABLES")
    print("="*60)
    
    # Run independent sampling
    pf_indep = reliability_analysis_independent(n_samples=50000, threshold=20.0)
    
    # Run D-vine copula on independent data
    pf_dvine_indep = reliability_analysis_independent_dvine(n_samples=50000, n_train = 5000, threshold=20.0)
    
    # Compare results for independent case
    print("\nCOMPARISON (Independent Variables):")
    print(f"Independent sampling: pf = {pf_indep:.6f}")
    print(f"D-vine copula:        pf = {pf_dvine_indep:.6f}")
    print(f"Difference:           {abs(pf_dvine_indep - pf_indep):.6f}")
    
    if abs(pf_dvine_indep - pf_indep) < 0.001:
        print("OK: Results match (within tolerance)")
    else:
        print("NOT OK: Results differ significantly")
    
    # Test 2: Correlated variables
    print("\n" + "="*60)
    print("TEST 2: CORRELATED VARIABLES")
    print("="*60)
    
    # Run D-vine copula on correlated data
    pf_dvine_corr = reliability_analysis_correlated_dvine(n_samples=50000, n_train = 5000, threshold=20.0)
    
    # Compare with independent
    print("\nCOMPARISON (Correlated vs Independent):")
    print(f"Independent variables: pf = {pf_indep:.6f}")
    print(f"Correlated variables:  pf = {pf_dvine_corr:.6f}")
    print(f"Difference:            {abs(pf_dvine_corr - pf_indep):.6f}")
    
    if abs(pf_dvine_corr - pf_indep) > 0.001:
        print("OK: Correlation affects failure probability (as expected)")
    else:
        print("NOT OK: Correlation has minimal effect")
    
    return pf_dvine_corr

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during reliability analysis: {e}")
        raise
