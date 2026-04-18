"""
Marine Mammal Sighting Data Cleaner
Cleans CalCOFI sighting data for use in a whale forecasting model that
ingests ocean temperature, currents, salinity, and phytoplankton inputs.
"""

import pandas as pd
import numpy as np

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_excel("data.xlsx")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Rename columns ─────────────────────────────────────────────────────────
# Strip any non-ASCII chars (degree signs etc.) from col names before matching,
# then apply a fixed positional rename list that matches the 61 known columns.
CLEAN_NAMES = [
    "event_code", "cruise", "date_local", "time_local", "datetime_local",
    "gmt_offset", "datetime_utc", "lat_raw", "lon_raw",
    "latitude", "longitude",
    "observer_port", "observer_starboard", "obs_quality",
    "visibility_mi", "precipitation", "cloud_pct",
    "glare_bearing_left", "glare_bearing_right", "glare_quality",
    "wind_dir_deg", "wind_speed_kt", "beaufort", "swell_ft",
    "sighting_no", "init_observer", "cue",
    "ship_heading_deg", "sighting_bearing_deg",
    "bino_reticle", "distance_m", "sighting_method",
    "envelope_depth", "envelope_depth2", "envelope_width", "envelope_width2",
    "count_best", "count_min", "count_max", "calf_count",
    "species_code", "species_code2", "species_pct1", "species_pct2",
    "on_effort", "off_effort_code", "on_transect", "off_transect_code",
    "behavior_primary", "behavior_other1", "behavior_other2",
    "photos_taken", "photographer1", "camera1", "frame1_first", "frame1_last",
    "photographer2", "camera2", "frame2_first", "frame2_last",
    "comments",
]
assert len(CLEAN_NAMES) == len(df.columns), \
    f"Column count mismatch: {len(df.columns)} in file vs {len(CLEAN_NAMES)} expected"
df.columns = CLEAN_NAMES

# ── 3. Drop fully-null / irrelevant columns ───────────────────────────────────
drop_cols = [
    "lat_raw", "lon_raw",           # 100 % null
    "datetime_local", "date_local", "time_local", "gmt_offset",  # redundant with datetime_utc
    "bino_reticle", "distance_m", "sighting_method",             # sighting geometry, not needed
    "envelope_depth", "envelope_depth2", "envelope_width", "envelope_width2",
    "photos_taken", "photographer1", "camera1", "frame1_first", "frame1_last",
    "photographer2", "camera2", "frame2_first", "frame2_last",   # photo metadata
    "species_code2", "species_pct1", "species_pct2",             # secondary species rare
    "off_effort_code", "off_transect_code",
    "glare_bearing_left", "glare_bearing_right", "glare_quality",
    "behavior_other1", "behavior_other2",
    "init_observer", "observer_port", "observer_starboard",
    "sighting_no",
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 4. Parse datetime ─────────────────────────────────────────────────────────
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
df["date"] = df["datetime_utc"].dt.normalize()
df["year"]  = df["datetime_utc"].dt.year
df["month"] = df["datetime_utc"].dt.month
df["day_of_year"] = df["datetime_utc"].dt.dayofyear

# ── 5. Normalize categorical effort/transect flags ────────────────────────────
def normalize_onoff(series):
    s = series.astype(str).str.strip().str.upper()
    s = s.replace({"0N": "ON", "OM": "ON", "0FF": "OFF"})
    s = s.where(s.isin(["ON", "OFF"]), other=np.nan)
    return s

df["on_effort"]  = normalize_onoff(df["on_effort"])
df["on_transect"] = normalize_onoff(df["on_transect"])

# ── 6. Normalize observation quality ─────────────────────────────────────────
quality_map = {"G": "G", "F": "F", "P": "P", "E": "E", "UX": "UX"}
df["obs_quality"] = (
    df["obs_quality"].astype(str).str.strip().str.upper()
    .map(lambda v: quality_map.get(v, np.nan))
)

# ── 7. Normalize event codes ──────────────────────────────────────────────────
df["event_code"] = df["event_code"].astype(str).str.strip().str.upper()
# Map common sighting event to 'W' (whale/cetacean sighting)
event_norm = {
    "W": "W",       # whale sighting
    "SB": "SB",     # start behavior
    "SE": "SE",     # sighting end
    "ST": "ST",     # start transect
    "ET": "ET",     # end transect
    "P": "P",       # position / condition update
    "OO": "OO",     # off-effort observation
    "B": "B",       # beginning of watch
    "E": "E",       # end of watch
    "PINN": "PINN", # pinniped
    "C": "C",       # cetacean sighting
    "OB": "OB",
    "SP": "SP",
    "XT": "XT",
    "SH": "SH",
    "RT": "RT",
    "OC": "OC",
    "FISH": "FISH",
    "SHIP": "SHIP",
}
df["event_code"] = df["event_code"].map(lambda v: event_norm.get(v, v))

# ── 8. Normalize species codes ────────────────────────────────────────────────
df["species_code"] = df["species_code"].astype(str).str.strip().str.upper()
df["species_code"] = df["species_code"].replace("NAN", np.nan)

# Species taxonomy lookup (scientific + common names)
CETACEAN_CODES = {
    # Large baleen whales
    "BM": ("Balaenoptera musculus",   "Blue Whale",          "large_whale"),
    "BP": ("Balaenoptera physalus",   "Fin Whale",           "large_whale"),
    "MN": ("Megaptera novaeangliae",  "Humpback Whale",      "large_whale"),
    "BA": ("Balaenoptera acutorostrata", "Minke Whale",      "large_whale"),
    "ER": ("Eschrichtius robustus",   "Gray Whale",          "large_whale"),
    "EI": ("Eubalaena japonica",      "North Pacific Right Whale", "large_whale"),
    # Toothed whales
    "PM": ("Physeter macrocephalus",  "Sperm Whale",         "large_whale"),
    "OO": ("Orcinus orca",            "Killer Whale (Orca)", "large_whale"),
    "GM": ("Globicephala macrorhynchus", "Short-finned Pilot Whale", "large_whale"),
    "SB": ("Steno bredanensis",       "Rough-toothed Dolphin", "dolphin"),
    "ZACA": ("Ziphius cavirostris",   "Cuvier's Beaked Whale", "large_whale"),
    "DSP": ("Delphinus sp.",          "Delphinus species",   "dolphin"),
    # Dolphins
    "DD": ("Delphinus delphis",       "Common Dolphin",      "dolphin"),
    "DC": ("Delphinus capensis",      "Long-beaked Common Dolphin", "dolphin"),
    "TT": ("Tursiops truncatus",      "Bottlenose Dolphin",  "dolphin"),
    "GG": ("Grampus griseus",         "Risso's Dolphin",     "dolphin"),
    "LO": ("Lagenorhynchus obliquidens", "Pacific White-sided Dolphin", "dolphin"),
    "PD": ("Phocoenoides dalli",      "Dall's Porpoise",     "dolphin"),
    "PC": ("Phocoena phocoena",       "Harbor Porpoise",     "dolphin"),
    "AT": ("Stenella attenuata",      "Spotted Dolphin",     "dolphin"),
    "LB": ("Lissodelphis borealis",   "Northern Right Whale Dolphin", "dolphin"),
    "SC": ("Stenella coeruleoalba",   "Striped Dolphin",     "dolphin"),
    "CU": ("Cephalorhynchus sp.",     "Cephalorhynchus sp.", "dolphin"),
    "CC": ("Cephalorhynchus commersonii", "Commerson's Dolphin", "dolphin"),
    "EL": ("Eulagenorhinchus sp.",    "Eulagenorhinchus sp.", "dolphin"),
    "MA": ("Mesoplodon sp.",          "Mesoplodon sp.",      "large_whale"),
    # Unidentified cetaceans
    "ULW":         (None, "Unidentified Large Whale",    "large_whale"),
    "USW":         (None, "Unidentified Small Whale",    "large_whale"),
    "UD":          (None, "Unidentified Dolphin",        "dolphin"),
    "UDF":         (None, "Unidentified Dolphin",        "dolphin"),
    "UT":          (None, "Unidentified Toothed Whale",  "large_whale"),
    "UNID_CETAC":  (None, "Unidentified Cetacean",       "cetacean_unid"),
    "UNID CETAC":  (None, "Unidentified Cetacean",       "cetacean_unid"),
    "UNIDCETAC":   (None, "Unidentified Cetacean",       "cetacean_unid"),
    "UNSMCET":     (None, "Unidentified Small Cetacean", "cetacean_unid"),
    "UNZIPH":      (None, "Unidentified Ziphiid",        "large_whale"),
    "USM":         (None, "Unidentified Small Mammal",   "cetacean_unid"),
    "UNOT":        (None, "Unidentified Other",          "cetacean_unid"),
    "UNSMWHALE":   (None, "Unidentified Small Whale",    "large_whale"),
    "UNID_CETAC":  (None, "Unidentified Cetacean",       "cetacean_unid"),
}

PINNIPED_CODES = {
    "PV": ("Phoca vitulina",          "Harbor Seal",         "pinniped"),
    "UP": ("Unidentified pinniped",   "Unidentified Pinniped", "pinniped"),
    "UPINN": ("Unidentified pinniped","Unidentified Pinniped", "pinniped"),
    "UNIDPINN": ("Unidentified pinniped", "Unidentified Pinniped", "pinniped"),
    "UNID PINN": ("Unidentified pinniped", "Unidentified Pinniped", "pinniped"),
    "UNID_PINN": ("Unidentified pinniped", "Unidentified Pinniped", "pinniped"),
    "UNIDP":    ("Unidentified pinniped", "Unidentified Pinniped", "pinniped"),
    "UNIDFS":   ("Unidentified fur seal", "Unidentified Fur Seal", "pinniped"),
    "NFS":      ("Callorhinus ursinus",   "Northern Fur Seal",    "pinniped"),
    "NF":       ("Callorhinus ursinus",   "Northern Fur Seal",    "pinniped"),
    "PINN":     ("Unidentified pinniped", "Unidentified Pinniped", "pinniped"),
    "FUR SEAL": ("Callorhinus sp.",       "Fur Seal",             "pinniped"),
}

OTHER_CODES = {
    "MOLA MOLA": ("Mola mola",  "Ocean Sunfish",  "fish"),
    "T":          (None,        "Turtle",         "turtle"),
    "TURT":       (None,        "Turtle",         "turtle"),
    "TURTLE":     (None,        "Turtle",         "turtle"),
    "TURTLEHAWKS BILL": (None,  "Hawksbill Turtle", "turtle"),
    "SHIP":       (None,        "Ship",           "vessel"),
    "NAVY":       (None,        "Navy vessel",    "vessel"),
    "DSP":        ("Delphinus sp.", "Delphinus sp.", "dolphin"),
    "DSPPP.":     ("Delphinus sp.", "Delphinus sp.", "dolphin"),
}

ALL_TAXA = {**CETACEAN_CODES, **PINNIPED_CODES, **OTHER_CODES}

# Also normalize common alternate codes
SPECIES_ALIASES = {
    "BBO/BE": "BBO",
    "BBO":    "BBO",  # keep as-is if unknown
    "DSP":    "DSP",
    "DSPP.":  "DSP",
    "ZACA":   "ZACA",
    "ZICA":   "ZACA",
    "UNID_OBJCT": None,
    "PP":     "PD",  # Phocoena phocoena -> treat same as harbor porpoise
}

def normalize_species(code):
    if pd.isna(code) or code in ("NAN", ""):
        return np.nan
    code = str(code).strip().upper()
    return SPECIES_ALIASES.get(code, code)

df["species_code"] = df["species_code"].apply(normalize_species)

# Add taxonomy columns
def get_taxon(code, idx):
    if pd.isna(code):
        return np.nan
    entry = ALL_TAXA.get(code)
    if entry:
        return entry[idx]
    return np.nan

df["species_scientific"] = df["species_code"].apply(lambda c: get_taxon(c, 0))
df["species_common"]     = df["species_code"].apply(lambda c: get_taxon(c, 1))
df["species_group"]      = df["species_code"].apply(lambda c: get_taxon(c, 2))

# ── 9. Numeric conversions ────────────────────────────────────────────────────
def to_numeric_col(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")

for col in ["visibility_mi", "cloud_pct", "wind_speed_kt", "beaufort",
            "count_best", "count_min", "count_max", "calf_count",
            "ship_heading_deg", "sighting_bearing_deg"]:
    if col in df.columns:
        df[col] = to_numeric_col(df[col])

# ── 10. Boolean flags ─────────────────────────────────────────────────────────
df["is_on_effort"]  = df["on_effort"]  == "ON"
df["is_on_transect"] = df["on_transect"] == "ON"
df["is_sighting"]   = df["species_code"].notna()
df["is_cetacean"]   = df["species_group"].isin(["large_whale", "dolphin", "cetacean_unid"])
df["is_large_whale"] = df["species_group"] == "large_whale"

# ── 11. Drop redundant string flag cols (now have booleans) ──────────────────
df.drop(columns=["on_effort", "on_transect"], inplace=True)

# ── 12. Reorder columns ───────────────────────────────────────────────────────
id_cols      = ["cruise", "event_code", "datetime_utc", "date", "year", "month", "day_of_year"]
location_cols = ["latitude", "longitude"]
env_cols     = ["obs_quality", "visibility_mi", "precipitation", "cloud_pct",
                "wind_dir_deg", "wind_speed_kt", "beaufort", "swell_ft"]
sighting_cols = ["is_sighting", "is_cetacean", "is_large_whale",
                 "species_code", "species_common", "species_scientific", "species_group",
                 "count_best", "count_min", "count_max", "calf_count",
                 "behavior_primary", "cue"]
effort_cols  = ["is_on_effort", "is_on_transect"]
nav_cols     = ["ship_heading_deg", "sighting_bearing_deg"]
misc_cols    = ["comments"]

ordered = (id_cols + location_cols + env_cols + sighting_cols +
           effort_cols + nav_cols + misc_cols)
remaining = [c for c in df.columns if c not in ordered]
df = df[ordered + remaining]

# ── 13. Summary stats ─────────────────────────────────────────────────────────
print(f"\nCleaned shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nDate range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
print(f"Cruises: {df['cruise'].nunique()}")
print(f"Total sightings: {df['is_sighting'].sum()}")
print(f"Cetacean sightings: {df['is_cetacean'].sum()}")
print(f"Large whale sightings: {df['is_large_whale'].sum()}")
print(f"\nOn-effort records: {df['is_on_effort'].sum()}")
print(f"On-effort cetacean sightings: {(df['is_on_effort'] & df['is_cetacean']).sum()}")
print(f"\nTop whale species (on-effort):")
on_eff = df[df["is_on_effort"] & df["is_large_whale"]]
print(on_eff.groupby(["species_code", "species_common"])["count_best"].agg(["count","sum"]).rename(columns={"count":"sightings","sum":"total_best"}).sort_values("sightings", ascending=False).head(15))
print(f"\nNull counts in key columns:")
key_cols = ["datetime_utc","latitude","longitude","species_code","count_best","is_on_effort","obs_quality"]
print(df[key_cols].isnull().sum())

# ── 14. Save ──────────────────────────────────────────────────────────────────
out_path = "cleaned_data.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Also save a sightings-only subset for quick model use
sightings_df = df[df["is_cetacean"] & df["is_on_effort"]].copy()
sightings_df.to_csv("cetacean_sightings.csv", index=False)
print(f"Saved: cetacean_sightings.csv ({len(sightings_df)} on-effort cetacean sightings)")
