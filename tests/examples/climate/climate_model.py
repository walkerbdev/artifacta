import numpy as np
import xarray as xr


class ClimateModel:
    def __init__(self, resolution="1deg", ensemble_size=10, start_year=2010, end_year=2020):
        self.resolution = resolution
        self.ensemble_size = ensemble_size
        self.start_year = start_year
        self.end_year = end_year

    def load_observations(self):
        """Load historical temperature observations"""
        obs_data = xr.open_dataset("observations.nc")
        return obs_data

    def run_ensemble(self, forcing_scenario="RCP4.5"):
        """Run ensemble of climate simulations"""
        results = []

        for member in range(self.ensemble_size):
            # Initialize ensemble member with perturbed initial conditions
            np.random.seed(42 + member)
            initial_perturbation = np.random.normal(0, 0.1)

            # Run climate model
            temperature_anomaly = self.simulate_temperature(forcing_scenario, initial_perturbation)
            results.append(temperature_anomaly)

        # Calculate ensemble mean and spread
        ensemble_mean = np.mean(results, axis=0)
        ensemble_std = np.std(results, axis=0)

        return {"mean": ensemble_mean, "std": ensemble_std, "members": results}

    def simulate_temperature(self, forcing, perturbation):
        """Simple energy balance model for temperature simulation"""
        years = range(self.start_year, self.end_year + 1)

        # Simple energy balance model
        temperature = []
        temp = perturbation

        for year in years:
            # Forcing increases linearly
            forcing_value = 0.5 + (year - self.start_year) * 0.05

            # Temperature response with feedback
            delta_temp = 0.7 * forcing_value - 0.1 * temp
            temp += delta_temp
            temperature.append(temp)

        return np.array(temperature)


# Example usage
if __name__ == "__main__":
    model = ClimateModel(resolution="1deg", ensemble_size=10)
    projection = model.run_ensemble(forcing_scenario="RCP4.5")
    print(f"Projected warming: {projection['mean'][-1]:.2f} ± {projection['std'][-1]:.2f}°C")
