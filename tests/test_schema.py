"""Column-name harmonization: merge by NAME across files, never by position."""
from src.schema import RENAME_MAP, CANONICAL_COLS, UNIT_NOTES, norm


def test_pm_variants_map_to_canonical():
    assert RENAME_MAP[norm("PM2.5")] == "pm25"
    assert RENAME_MAP[norm("PM10")] == "pm10"


def test_met_variants_map_across_files():
    # WS/Wind Speed, WD/Wind Dir, Temp/Temperature, SR/Solar Rad, VWS/V Wind Speed
    assert RENAME_MAP[norm("WS")] == RENAME_MAP[norm("Wind Speed")] == "wind_speed"
    assert RENAME_MAP[norm("WD")] == RENAME_MAP[norm("Wind Dir")] == "wind_dir"
    assert RENAME_MAP[norm("Temp")] == RENAME_MAP[norm("Temperature")] == "temp"
    assert RENAME_MAP[norm("SR")] == RENAME_MAP[norm("Solar Rad")] == "solar_rad"
    assert RENAME_MAP[norm("VWS")] == RENAME_MAP[norm("V Wind Speed")] == "v_wind_speed"
    assert RENAME_MAP[norm("Ratio")] == RENAME_MAP[norm("RS")] == "ratio"


def test_norm_is_case_and_space_insensitive():
    assert norm("  Wind   Speed ") == "wind speed"


def test_co_is_ppm_others_ppb():
    assert UNIT_NOTES["co"] == "ppm"
    assert UNIT_NOTES["so2"] == "ppb"


def test_canonical_order_has_pm25_before_pm10():
    # regardless of raw file order, the canonical schema is fixed
    assert CANONICAL_COLS.index("pm25") < CANONICAL_COLS.index("pm10")
