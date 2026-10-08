import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DEPTHS = {
    "soil_surface_temperature_C": 0.00,
    "soil_temp_5cm_C": 0.05,
    "soil_temp_10cm_C": 0.10,
    "soil_temp_20cm_C": 0.20,
}


# ============================================================
# INPUT / DATA PREPARATION
# ============================================================

def load_weather_data(input_file):
    """Load and validate the weather CSV."""

    required_columns = {
        "date_time_local",
        "temperature",
        "solar_radiation",
        "precipitation",
    }

    df = pd.read_csv(input_file)

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


def parse_vancouver_datetime(df):
    """
    Parse timestamps such as:

        2026-10-07 18:00:00 PDT

    and create timezone-aware Vancouver and UTC timestamps.
    """

    raw_datetime = (
        df["date_time_local"]
        .astype(str)
        .str.strip()
    )

    # Remove PDT/PST before timezone localization.
    datetime_without_abbreviation = (
        raw_datetime
        .str.replace(
            r"\s+(PDT|PST)$",
            "",
            regex=True,
        )
    )

    local_naive = pd.to_datetime(
        datetime_without_abbreviation,
        errors="coerce",
    )

    if local_naive.isna().any():

        bad_rows = df.loc[
            local_naive.isna(),
            "date_time_local",
        ].head()

        raise ValueError(
            "Unable to parse datetime values:\n"
            + "\n".join(map(str, bad_rows))
        )

    # Vancouver's timezone automatically handles PST/PDT.
    datetime_local = local_naive.dt.tz_localize(
        "America/Vancouver",
        ambiguous="infer",
        nonexistent="shift_forward",
    )

    datetime_utc = datetime_local.dt.tz_convert("UTC")

    result = df.copy()

    result["_datetime_local"] = datetime_local
    result["_datetime_utc"] = datetime_utc

    return result


def prepare_weather_data(df):
    """Sort the data and prepare the numeric weather variables."""

    df = (
        df
        .sort_values("_datetime_utc")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "temperature",
        "solar_radiation",
        "precipitation",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Temperature:
    # interpolate short gaps and then fill any remaining
    # leading/trailing gaps.
    df["temperature"] = (
        df["temperature"]
        .interpolate(limit=6)
        .ffill()
        .bfill()
    )

    # Missing radiation is treated as zero.
    df["solar_radiation"] = (
        df["solar_radiation"]
        .fillna(0)
        .clip(lower=0)
    )

    # Missing precipitation is treated as zero.
    df["precipitation"] = (
        df["precipitation"]
        .fillna(0)
        .clip(lower=0)
    )

    if df["temperature"].isna().any():
        raise ValueError(
            "Temperature data contains unresolvable missing values."
        )

    return df


def calculate_timestep(df):
    """Return the median input timestep in seconds."""

    time_difference = (
        df["_datetime_utc"]
        .diff()
        .dt.total_seconds()
    )

    median_dt = time_difference.dropna().median()

    if not np.isfinite(median_dt):
        raise ValueError(
            "Unable to determine input timestep."
        )

    if median_dt <= 0:
        raise ValueError(
            "Datetime values must be strictly increasing."
        )

    return median_dt


# ============================================================
# SOIL MODEL SETUP
# ============================================================

def create_soil_grid(
    model_depth=2.0,
    dz=0.01,
):
    """
    Create the vertical soil grid.

    Returns
    -------
    depths : numpy.ndarray
        Depth of each numerical soil layer in metres.
    """

    if model_depth <= 0:
        raise ValueError(
            "model_depth must be greater than zero."
        )

    if dz <= 0:
        raise ValueError(
            "dz must be greater than zero."
        )

    return np.arange(
        0,
        model_depth + dz,
        dz,
    )


def get_output_indices(
    depths,
    output_depths=OUTPUT_DEPTHS,
):
    """
    Find the numerical grid index corresponding to each
    requested output depth.
    """

    indices = {}

    for column_name, depth in output_depths.items():

        index = int(
            np.argmin(
                np.abs(depths - depth)
            )
        )

        actual_depth = depths[index]

        # Make sure the requested depth is actually represented
        # closely enough by the numerical grid.
        if not np.isclose(
            actual_depth,
            depth,
            atol=1e-9,
        ):
            raise ValueError(
                f"Could not represent depth {depth} m."
            )

        indices[column_name] = index

    return indices


def estimate_initial_soil_temperature(
    df,
    default_temperature=10.0,
    initial_period_hours=24 * 7,
):
    """
    Estimate the initial soil temperature from the first week
    of air-temperature observations.
    """

    samples = min(
        len(df),
        initial_period_hours,
    )

    initial_temperature = (
        df["temperature"]
        .iloc[:samples]
        .mean()
    )

    if not np.isfinite(initial_temperature):
        return default_temperature

    return float(initial_temperature)


def initialize_soil_profile(
    number_of_layers,
    initial_temperature,
):
    """Create the initial soil temperature profile."""

    return np.full(
        number_of_layers,
        initial_temperature,
        dtype=float,
    )


# ============================================================
# SURFACE ENERGY MODEL
# ============================================================

def calculate_absorbed_solar_radiation(
    solar_radiation,
    albedo,
    solar_absorptivity,
):
    """
    Calculate solar radiation treated as available to heat
    the soil surface.

    Parameters
    ----------
    solar_radiation : float
        Incoming solar radiation in W/m².

    albedo : float
        Reflected fraction of radiation.

    solar_absorptivity : float
        Fraction of non-reflected radiation contributing to
        the surface energy budget.
    """

    return (
        solar_radiation
        * (1.0 - albedo)
        * solar_absorptivity
    )


def calculate_solar_temperature_change(
    absorbed_solar,
    dt,
    volumetric_heat_capacity,
    active_depth,
    solar_soil_fraction,
):
    """
    Convert absorbed solar energy into an approximate
    temperature increment of the near-surface soil.
    """

    # W/m² × seconds = J/m²
    solar_energy = (
        absorbed_solar
        * dt
    )

    heat_capacity = (
        volumetric_heat_capacity
        * active_depth
    )

    temperature_change = (
        solar_energy
        * solar_soil_fraction
        / heat_capacity
    )

    # Prevent anomalous radiation values from creating
    # implausibly large single-timestep temperature changes.
    return float(
        np.clip(
            temperature_change,
            0,
            8,
        )
    )


def calculate_surface_temperature(
    air_temperature,
    solar_radiation,
    precipitation,
    previous_surface_temperature,
    dt,
    albedo,
    solar_absorptivity,
    solar_soil_fraction,
    volumetric_heat_capacity,
    active_depth,
    surface_air_weight,
    moisture_surface_factor,
):
    """
    Calculate the effective soil surface temperature for one
    timestep.
    """

    absorbed_solar = (
        calculate_absorbed_solar_radiation(
            solar_radiation=solar_radiation,
            albedo=albedo,
            solar_absorptivity=solar_absorptivity,
        )
    )

    solar_temperature_change = (
        calculate_solar_temperature_change(
            absorbed_solar=absorbed_solar,
            dt=dt,
            volumetric_heat_capacity=(
                volumetric_heat_capacity
            ),
            active_depth=active_depth,
            solar_soil_fraction=(
                solar_soil_fraction
            ),
        )
    )

    target_temperature = (
        air_temperature
        + solar_temperature_change
    )

    # Rainfall is used as a crude proxy for increased soil
    # moisture and therefore increased thermal inertia.
    if precipitation > 2.0:
        air_weight = (
            surface_air_weight
            * moisture_surface_factor
        )
    else:
        air_weight = surface_air_weight

    surface_temperature = (
        previous_surface_temperature
        + air_weight
        * (
            target_temperature
            - previous_surface_temperature
        )
    )

    return float(surface_temperature)


# ============================================================
# SOIL THERMAL PROPERTIES
# ============================================================

def estimate_effective_thermal_diffusivity(
    base_diffusivity,
    precipitation,
):
    """
    Estimate effective thermal diffusivity for the current
    timestep.

    This is deliberately simple. Actual soil moisture data
    would be preferable to using precipitation as a proxy.
    """

    if precipitation > 5.0:
        return base_diffusivity * 0.85

    return base_diffusivity


# ============================================================
# NUMERICAL SOLVER
# ============================================================

def calculate_substeps(
    dt,
    dz,
    thermal_diffusivity,
):
    """
    Determine how many numerical substeps are required for
    stability of the explicit finite-difference solver.
    """

    maximum_stable_dt = (
        0.45
        * dz**2
        / thermal_diffusivity
    )

    return max(
        1,
        int(
            np.ceil(
                dt / maximum_stable_dt
            )
        ),
    )


def propagate_heat(
    soil_temperature,
    surface_temperature,
    thermal_diffusivity,
    dt,
    dz,
):
    """
    Advance the soil-temperature profile by one timestep.

    Uses the one-dimensional heat equation:

        dT/dt = alpha * d²T/dz²

    with:

        - fixed surface temperature
        - zero-gradient bottom boundary
    """

    number_of_substeps = calculate_substeps(
        dt=dt,
        dz=dz,
        thermal_diffusivity=thermal_diffusivity,
    )

    sub_dt = (
        dt / number_of_substeps
    )

    coefficient = (
        thermal_diffusivity
        * sub_dt
        / dz**2
    )

    temperature = soil_temperature.copy()

    for _ in range(number_of_substeps):

        new_temperature = temperature.copy()

        # Surface boundary condition.
        new_temperature[0] = (
            surface_temperature
        )

        # Interior layers.
        new_temperature[1:-1] = (
            temperature[1:-1]
            + coefficient
            * (
                temperature[2:]
                - 2.0 * temperature[1:-1]
                + temperature[:-2]
            )
        )

        # Bottom boundary:
        # zero temperature gradient.
        new_temperature[-1] = (
            temperature[-1]
            + coefficient
            * (
                temperature[-2]
                - temperature[-1]
            )
        )

        temperature = new_temperature

    return temperature


# ============================================================
# OUTPUT
# ============================================================

def extract_soil_temperatures(
    soil_temperature,
    output_indices,
):
    """Extract the temperatures at the requested depths."""

    return {
        column_name: float(
            soil_temperature[index]
        )
        for column_name, index
        in output_indices.items()
    }


def build_output_dataframe(
    df,
    air_temperatures,
    soil_temperature_results,
):
    """Build the final user-facing output DataFrame."""

    output = pd.DataFrame({
        "datetime_local": (
            df["_datetime_local"]
            .dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        ),

        "datetime_utc": (
            df["_datetime_utc"]
            .dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        ),

        "air_temperature_C": air_temperatures,
    })

    for column_name in OUTPUT_DEPTHS:
        output[column_name] = (
            soil_temperature_results[column_name]
        )

    # Round all temperature estimates to 2 decimal places.
    temperature_columns = [
        "air_temperature_C",
        *OUTPUT_DEPTHS.keys(),
    ]

    output[temperature_columns] = (
        output[temperature_columns]
        .round(2)
    )

    return output


def save_output(
    output,
    output_file,
):
    """Write the final dataset to CSV."""

    output.to_csv(
        output_file,
        index=False,
    )


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def estimate_soil_temperature(
    input_file,
    output_file,
    thermal_diffusivity=0.8e-6,
    volumetric_heat_capacity=2.2e6,
    albedo=0.20,
    solar_absorptivity=0.75,
    solar_soil_fraction=0.35,
    surface_air_weight=0.20,
    moisture_surface_factor=0.70,
    active_depth=0.05,
    model_depth=2.0,
    dz=0.01,
    default_initial_temperature=10.0,
):
    """
    Estimate hourly Vancouver soil temperatures.

    This is the main orchestration function. Individual stages
    of the model are implemented in separate functions so that
    they can be tested or replaced independently.

    Returns
    -------
    pandas.DataFrame
        The generated soil-temperature dataset.
    """

    # --------------------------------------------------------
    # 1. Load and prepare weather data
    # --------------------------------------------------------

    df = load_weather_data(
        input_file
    )

    df = parse_vancouver_datetime(
        df
    )

    df = prepare_weather_data(
        df
    )

    median_dt = calculate_timestep(
        df
    )

    if median_dt != 3600:
        raise ValueError(
            "Input data must be hourly."
        )


    # --------------------------------------------------------
    # 2. Create soil grid
    # --------------------------------------------------------

    depths = create_soil_grid(
        model_depth=model_depth,
        dz=dz,
    )

    output_indices = get_output_indices(
        depths
    )


    # --------------------------------------------------------
    # 3. Initialize soil
    # --------------------------------------------------------

    initial_temperature = (
        estimate_initial_soil_temperature(
            df,
            default_temperature=(
                default_initial_temperature
            ),
        )
    )

    soil_temperature = (
        initialize_soil_profile(
            number_of_layers=len(depths),
            initial_temperature=(
                initial_temperature
            ),
        )
    )

    previous_surface_temperature = (
        initial_temperature
    )


    # --------------------------------------------------------
    # 4. Prepare result storage
    # --------------------------------------------------------

    soil_temperature_results = {
        column_name: []
        for column_name in OUTPUT_DEPTHS
    }

    air_temperatures = []


    # --------------------------------------------------------
    # 5. Run thermal model
    # --------------------------------------------------------

    for i in range(len(df)):

        air_temperature = float(
            df.loc[i, "temperature"]
        )

        solar_radiation = float(
            df.loc[i, "solar_radiation"]
        )

        precipitation = float(
            df.loc[i, "precipitation"]
        )


        # Determine actual timestep.

        if i == 0:

            dt = median_dt

        else:

            dt = (
                df.loc[i, "_datetime_utc"]
                - df.loc[i - 1, "_datetime_utc"]
            ).total_seconds()

            if not np.isfinite(dt) or dt <= 0:
                dt = median_dt


        # ----------------------------------------------------
        # Surface temperature
        # ----------------------------------------------------

        surface_temperature = (
            calculate_surface_temperature(
                air_temperature=air_temperature,
                solar_radiation=solar_radiation,
                precipitation=precipitation,
                previous_surface_temperature=(
                    previous_surface_temperature
                ),
                dt=dt,
                albedo=albedo,
                solar_absorptivity=(
                    solar_absorptivity
                ),
                solar_soil_fraction=(
                    solar_soil_fraction
                ),
                volumetric_heat_capacity=(
                    volumetric_heat_capacity
                ),
                active_depth=active_depth,
                surface_air_weight=(
                    surface_air_weight
                ),
                moisture_surface_factor=(
                    moisture_surface_factor
                ),
            )
        )

        previous_surface_temperature = (
            surface_temperature
        )


        # ----------------------------------------------------
        # Soil thermal properties
        # ----------------------------------------------------

        effective_diffusivity = (
            estimate_effective_thermal_diffusivity(
                base_diffusivity=(
                    thermal_diffusivity
                ),
                precipitation=precipitation,
            )
        )


        # ----------------------------------------------------
        # Propagate heat through soil
        # ----------------------------------------------------

        soil_temperature = propagate_heat(
            soil_temperature=soil_temperature,
            surface_temperature=surface_temperature,
            thermal_diffusivity=(
                effective_diffusivity
            ),
            dt=dt,
            dz=dz,
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        air_temperatures.append(
            air_temperature
        )

        temperatures = (
            extract_soil_temperatures(
                soil_temperature=soil_temperature,
                output_indices=output_indices,
            )
        )

        for column_name, temperature in (
            temperatures.items()
        ):

            soil_temperature_results[
                column_name
            ].append(temperature)


    # --------------------------------------------------------
    # 6. Build output
    # --------------------------------------------------------

    output = build_output_dataframe(
        df=df,
        air_temperatures=air_temperatures,
        soil_temperature_results=(
            soil_temperature_results
        ),
    )


    # --------------------------------------------------------
    # 7. Save output
    # --------------------------------------------------------

    save_output(
        output=output,
        output_file=output_file,
    )


    return output


# ============================================================
# EXAMPLE
# ============================================================


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_DIR / "data"

    INPUT_FILE = DATA_DIR / "vancouver_climate_hourly_2025.csv"
    OUTPUT_FILE = DATA_DIR / "vancouver_soil_temperature.csv"

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    result = estimate_soil_temperature(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
    )

    print(
        result.head()
    )
