from sim.teg import TEG

import csv
from datetime import datetime
from collections import defaultdict
from pathlib import Path


def calculate_daily_energy(
    csv_file,
    teg: TEG,
    soil_column="soil_temp_10cm_C"
):
    """
    Read weather/soil CSV line-by-line and calculate Seebeck power
    between consecutive measurements.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.

    soil_column : str
        Soil temperature column to use as the cold side.

    Returns
    -------
    daily_energy : dict
        Dictionary mapping date -> energy in Joules.
    """

    daily_energy = defaultdict(float)

    previous_row_data = None

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            # Parse UTC timestamp.
            # Remove " UTC" because datetime.strptime doesn't need it.
            timestamp = datetime.strptime(
                row["datetime_utc"].replace(" UTC", ""),
                "%Y-%m-%d %H:%M:%S"
            )

            air_temp = float(row["air_temperature_C"])
            soil_temp = float(row[soil_column])

            current_row_data = {
                "timestamp": timestamp,
                "air_temp": air_temp,
                "soil_temp": soil_temp,
            }

            # Need two measurements to calculate an interval.
            if previous_row_data is not None:

                dt_hours = (
                    timestamp - previous_row_data["timestamp"]
                ).total_seconds() / 3600.0

                # Temperature difference
                delta_T = (
                    previous_row_data["air_temp"] - previous_row_data["soil_temp"]
                )

                # Maximum power transfer
                power_watts = teg.maximum_power(
                    previous_row_data["air_temp"],
                    previous_row_data["soil_temp"]
                )

                # Energy produced during this interval in joules
                energy_joules = power_watts * dt_hours * 3600.0

                # Assign interval to the day on which it ends
                day = timestamp.date()

                daily_energy[day] += energy_joules

            previous_row_data = current_row_data

    return dict(daily_energy)

if __name__ == "__main__":

    # Verify that the input file exists
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_DIR / "data"

    INPUT_FILE = DATA_DIR / "vancouver_soil_temperature.csv"

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Calculate daily energy using the TEG model
    daily_energy = calculate_daily_energy(
        INPUT_FILE,
        teg=TEG(seebeck_coefficient=0.0129, internal_resistance=4.0), # TEC1-12706 characteristics
        soil_column="soil_temp_10cm_C"
    )

    for day, energy in daily_energy.items():
        print(f"{day}: {energy:.3f} Joules")


    import matplotlib.pyplot as plt

    dates = list(daily_energy.keys())
    energy = list(daily_energy.values())

    plt.figure(figsize=(12, 5))
    plt.plot(dates, energy, marker="o", linewidth=1)

    plt.xlabel("Date")
    plt.ylabel("Energy (Joules)")
    plt.title("Daily TEG Energy Production")

    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()