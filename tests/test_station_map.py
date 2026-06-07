"""Station canonicalization: spelling merges only; ambiguous sites stay SEPARATE."""
from src.schema import canonical_station, AMBIGUOUS_STATIONS


def test_spelling_variants_merge():
    assert canonical_station("Mymensing")[0] == "Mymensingh"
    assert canonical_station("Mymensingh")[0] == "Mymensingh"
    for v in ("Narayonganj", "Narayangonj", "Narayanganj"):
        assert canonical_station(v)[0] == "Narayanganj"
    for v in ("TV center", "TV Center", "TV Sation"):
        assert canonical_station(v)[0] == "TV_Center"


def test_ambiguous_chittagong_sites_not_merged():
    # TV st-Chittagong must NOT collapse into TV_Center
    assert canonical_station("TV st-Chittagong")[0] == "TV_st_Chittagong"
    assert canonical_station("TV st-Chittagong")[0] != canonical_station("TV center")[0]
    # Agrabad Chittagong must NOT collapse into CDA
    assert canonical_station("Agrabad Chittagong")[0] == "Agrabad_Chittagong"
    assert canonical_station("Agrabad Chittagong")[0] != canonical_station("CDA")[0]
    # all four are listed as ambiguous for human sign-off
    for s in ("TV_Center", "TV_st_Chittagong", "Agrabad_Chittagong", "CDA"):
        assert s in AMBIGUOUS_STATIONS


def test_unknown_station_passes_through_flagged():
    canon, is_mapped = canonical_station("Some New Site")
    assert is_mapped is False
    assert canon == "Some New Site"  # kept, not dropped
