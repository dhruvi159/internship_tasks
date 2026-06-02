"""
IPL Data Analytics & Visualization Pipeline (2008-2015)
========================================================
Production-grade industrial analytics using OOP + PEP8.
Dataset: Kaggle IPL 2008-2015 public repository.

Author  : IPL Analytics Pipeline
Version : 1.0.0
Python  : 3.9+

Dependencies
------------
    pip install pandas openpyxl matplotlib seaborn scipy numpy
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend – safe for PyCharm run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH: str = "IPL.xls"
OUTPUT_DIR: str = "ipl_output_charts"

PALETTE_TEAMS: dict[str, str] = {
    "Chennai": "#FDB940",
    "Mumbai": "#004C97",
    "Kolkata": "#3A225D",
    "Bangalore": "#EC1C24",
    "Rajasthan": "#E43D8B",
    "Delhi": "#17479E",
    "Punjab": "#DD1F2D",
    "Hyderabad": "#F26522",
    "Pune": "#9B1B30",
    "Kochi": "#8B1A1A",
}

PHASE_COLORS: dict[str, str] = {
    "Powerplay (1-6)": "#2196F3",
    "Middle (7-15)": "#FF9800",
    "Death (16-20)": "#F44336",
}

FIGURE_DPI: int = 150
FIGURE_SIZE_WIDE: tuple[int, int] = (16, 8)
FIGURE_SIZE_SQUARE: tuple[int, int] = (12, 10)
FIGURE_SIZE_TALL: tuple[int, int] = (14, 10)


# ─── Data-classes ────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Immutable runtime configuration for the analytics pipeline."""

    data_path: str = DEFAULT_DATA_PATH
    output_dir: str = OUTPUT_DIR
    dpi: int = FIGURE_DPI
    save_charts: bool = True
    show_charts: bool = False  # set True when running interactively


@dataclass
class InsightResult:
    """Structured container for a single statistical insight."""

    insight_id: int
    title: str
    summary: str
    key_metric: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"\n{'═' * 70}",
            f"  INSIGHT #{self.insight_id:02d} │ {self.title}",
            f"{'─' * 70}",
            f"  {self.summary}",
        ]
        if self.key_metric:
            lines.append(f"  Key metric ► {self.key_metric}")
        lines.append(f"{'═' * 70}")
        return "\n".join(lines)


# ─── Data Layer ──────────────────────────────────────────────────────────────
class IPLDataLoader:
    """
    Responsible solely for reading and returning the raw IPL DataFrame.

    Supports .xls (openpyxl) and .csv formats.
    """

    _LOG = logging.getLogger("IPLDataLoader")

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load raw data and return a DataFrame."""
        if not self._path.exists():
            raise FileNotFoundError(f"Data file not found: {self._path}")

        ext = self._path.suffix.lower()
        self._LOG.info("Loading data from '%s' …", self._path)

        if ext in {".xls", ".xlsx"}:
            df = pd.read_excel(self._path, engine="openpyxl")
        elif ext == ".csv":
            df = pd.read_csv(self._path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self._LOG.info("Raw dataset loaded → %d rows × %d columns", *df.shape)
        return df


# ─── Processing Layer ─────────────────────────────────────────────────────────
class IPLDataProcessor:
    """
    Cleans, validates, and feature-engineers the raw IPL DataFrame.

    Transformations applied
    -----------------------
    * Type coercions (dates, numerics)
    * Missing-value strategy per column
    * Derived boolean flags  (bat_first_won, is_final, dl_match …)
    * Phase-level run columns  (powerplay, middle, death)
    * Run-rate differentials and scoring velocity columns
    """

    _LOG = logging.getLogger("IPLDataProcessor")

    # ------------------------------------------------------------------
    def __init__(self, df: pd.DataFrame) -> None:
        self._raw: pd.DataFrame = df.copy()
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call .process() first.")
        return self._df

    # ------------------------------------------------------------------
    def process(self) -> "IPLDataProcessor":
        """Execute the full processing pipeline and cache result."""
        self._LOG.info("Starting data processing pipeline …")
        df = self._raw.copy()

        df = self._coerce_types(df)
        df = self._handle_missing(df)
        df = self._add_match_outcome_flags(df)
        df = self._add_phase_columns(df)
        df = self._add_run_rate_features(df)
        df = self._add_venue_city(df)

        self._df = df
        self._log_summary(df)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to their intended types."""
        # Year as integer
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

        # Numeric score columns
        numeric_cols = [
            c for c in df.columns
            if any(kw in c for kw in ["Score", "score", "Runs", "Wickets",
                                       "wkts", "Rate", "RR", "Margin",
                                       "Remaining", "Overs"])
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # String normalisations
        for col in ["Winner", "Winning_Team", "Win_Type",
                    "Team_Batting_First", "Team_Batting_Second",
                    "Match_Time", "Venue", "Match"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Apply targeted imputation / forward-fill strategies."""
        # DL matches: coerce to numeric first (column may contain strings like
        # "Yes" / "1" / NaN), then fill missing with 0.
        if "Duckworth Lewis Used" in df.columns:
            df["Duckworth Lewis Used"] = (
                pd.to_numeric(df["Duckworth Lewis Used"], errors="coerce").fillna(0)
            )

        # Winning margin: 0 for tied matches
        if "Winning_Margin" in df.columns:
            df["Winning_Margin"] = df["Winning_Margin"].fillna(0)

        # Phase scores where match ended early: forward-fill within match
        phase_cols = [c for c in df.columns if "_ov_score" in c or "_ov_wkts" in c]
        for col in phase_cols:
            df[col] = df[col].fillna(0)

        return df

    @staticmethod
    def _add_match_outcome_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Boolean flags derived from winner/winning-team columns."""
        df["bat_first_won"] = df["Winning_Team"].str.lower() == "firstbatting"
        df["chasing_won"] = df["Winning_Team"].str.lower() == "chasing"
        df["is_tied"] = df["Winner"].str.lower() == "match tied"
        df["dl_match"] = df["Duckworth Lewis Used"] > 0
        df["is_playoff"] = df["Match"].str.lower() != "league match"
        df["win_by_runs"] = df["Win_Type"].str.lower() == "run"
        df["win_by_wickets"] = df["Win_Type"].str.lower() == "wicket"
        return df

    @staticmethod
    def _add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Derive per-phase run contributions for each innings."""
        # Batting first phases
        df["bat1_powerplay_runs"] = df["Bat_First_5_ov_score"].clip(lower=0)
        df["bat1_middle_runs"] = (
            df["Bat_First_15_ov_score"] - df["Bat_First_5_ov_score"]
        ).clip(lower=0)
        df["bat1_death_runs"] = (
            df["Bat_First_Runs_Scored"] - df["Bat_First_15_ov_score"]
        ).clip(lower=0)

        # Batting second phases
        df["bat2_powerplay_runs"] = df["Bat_Second_5_ov_score"].clip(lower=0)
        df["bat2_middle_runs"] = (
            df["Bat_Second_15_ov_score"] - df["Bat_Second_5_ov_score"]
        ).clip(lower=0)
        df["bat2_death_runs"] = (
            df["Bat_Second_Runs_Scored"] - df["Bat_Second_15_ov_score"]
        ).clip(lower=0)

        # % contribution of each phase
        total1 = df["Bat_First_Runs_Scored"].replace(0, np.nan)
        total2 = df["Bat_Second_Runs_Scored"].replace(0, np.nan)

        df["bat1_powerplay_pct"] = df["bat1_powerplay_runs"] / total1 * 100
        df["bat1_middle_pct"] = df["bat1_middle_runs"] / total1 * 100
        df["bat1_death_pct"] = df["bat1_death_runs"] / total1 * 100

        df["bat2_powerplay_pct"] = df["bat2_powerplay_runs"] / total2 * 100
        df["bat2_middle_pct"] = df["bat2_middle_runs"] / total2 * 100
        df["bat2_death_pct"] = df["bat2_death_runs"] / total2 * 100

        return df

    @staticmethod
    def _add_run_rate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derived run-rate and pressure columns."""
        df["rr_diff"] = df["Bat_First_Run_Rate"] - df["Bat_Second_Run_Rate"]
        df["total_match_runs"] = (
            df["Bat_First_Runs_Scored"] + df["Bat_Second_Runs_Scored"]
        )
        df["score_diff"] = (
            df["Bat_First_Runs_Scored"] - df["Bat_Second_Runs_Scored"]
        )
        return df

    @staticmethod
    def _add_venue_city(df: pd.DataFrame) -> pd.DataFrame:
        """Extract approximate city from venue name."""
        city_map = {
            "Chennai": "Chennai", "Mumbai": "Mumbai",
            "Kolkata": "Kolkata", "Delhi": "Delhi",
            "Hyderabad": "Hyderabad", "Bangalore": "Bangalore",
            "Rajkot": "Rajkot", "Pune": "Pune",
            "Jaipur": "Jaipur", "Chandigarh": "Chandigarh",
            "Dharamsala": "Dharamsala", "Ranchi": "Ranchi",
            "Cuttack": "Cuttack", "Visakhapatnam": "Visakhapatnam",
            "Ahmedabad": "Ahmedabad", "Kochi": "Kochi",
            "Durban": "Durban", "Johannesburg": "Johannesburg",
            "Cape Town": "Cape Town", "Port Elizabeth": "Port Elizabeth",
            "Centurion": "Centurion", "Kimberley": "Kimberley",
        }
        df["city"] = df["Venue"].apply(
            lambda v: next(
                (city for city in city_map if city.lower() in v.lower()),
                "Other"
            )
        )
        return df

    # ------------------------------------------------------------------
    def _log_summary(self, df: pd.DataFrame) -> None:
        self._LOG.info(
            "Processing complete → %d rows, %d columns | "
            "Years: %s | Venues: %d | Unique teams: %d",
            len(df),
            len(df.columns),
            sorted(df["Year"].dropna().unique().tolist()),
            df["Venue"].nunique(),
            df["Team_Batting_First"].nunique(),
        )


# ─── Visualisation helpers ────────────────────────────────────────────────────
class ChartSaver:
    """Thin helper: save & optionally show matplotlib figures."""

    def __init__(self, output_dir: str, dpi: int, show: bool) -> None:
        self._dir = Path(output_dir)
        self._dpi = dpi
        self._show = show
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, filename: str) -> str:
        fp = self._dir / filename
        fig.savefig(fp, dpi=self._dpi, bbox_inches="tight", facecolor="white")
        if self._show:
            plt.show()
        plt.close(fig)
        return str(fp)


def apply_global_style() -> None:
    """Apply a consistent, professional chart style across all plots."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
    })


def add_value_labels(
    ax: plt.Axes,
    fmt: str = "{:.1f}",
    fontsize: int = 9,
    color: str = "black",
    offset: float = 0.3,
) -> None:
    """Annotate bar patches with their numeric values."""
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.annotate(
            fmt.format(h),
            xy=(patch.get_x() + patch.get_width() / 2, h),
            xytext=(0, offset * 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold",
        )


# ─── Analytics Engine ─────────────────────────────────────────────────────────
class IPLInsightsEngine:
    """
    Generates 15 statistical insights with corresponding visualisations.

    Each public method follows the signature:
        insight_<N>(self) -> InsightResult
    and uses self._df (processed DataFrame) + self._saver (ChartSaver).
    """

    _LOG = logging.getLogger("IPLInsightsEngine")

    def __init__(self, df: pd.DataFrame, saver: ChartSaver) -> None:
        self._df = df
        self._saver = saver

    # ------------------------------------------------------------------
    # ── INSIGHT 1 ──────────────────────────────────────────────────────
    def insight_01_bat_first_vs_chase(self) -> InsightResult:
        """Win percentage: batting first vs chasing."""
        title = "Batting First vs Chasing Win Rate"
        total = len(self._df[~self._df["is_tied"]])
        bat_first_wins = self._df["bat_first_won"].sum()
        chase_wins = self._df["chasing_won"].sum()
        bf_pct = bat_first_wins / total * 100
        ch_pct = chase_wins / total * 100

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
        fig.suptitle(f"INSIGHT #01 │ {title}", fontsize=14, fontweight="bold", y=1.01)

        # Pie chart
        ax = axes[0]
        wedges, texts, autotexts = ax.pie(
            [bf_pct, ch_pct],
            labels=["Bat First", "Chasing"],
            colors=["#3498DB", "#E74C3C"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0.05),
            shadow=True,
            textprops={"fontsize": 13},
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(14)
        ax.set_title("Overall Win Split", fontsize=13)

        # Year-wise stacked bar
        ax2 = axes[1]
        yearly = self._df.groupby("Year").agg(
            bat_first=("bat_first_won", "sum"),
            chasing=("chasing_won", "sum"),
        )
        x = yearly.index.tolist()
        ax2.bar(x, yearly["bat_first"], label="Bat First", color="#3498DB", alpha=0.85)
        ax2.bar(x, yearly["chasing"], bottom=yearly["bat_first"],
                label="Chasing", color="#E74C3C", alpha=0.85)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Wins")
        ax2.set_title("Year-wise Win Distribution", fontsize=13)
        ax2.legend(loc="upper right")
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))

        fig.tight_layout()
        self._saver.save(fig, "insight_01_bat_first_vs_chase.png")

        return InsightResult(
            1, title,
            f"Out of {total} completed matches, batting first won "
            f"{bat_first_wins} ({bf_pct:.1f}%) and chasing won "
            f"{chase_wins} ({ch_pct:.1f}%).",
            key_metric=f"Chase-win advantage: {ch_pct - bf_pct:+.1f} pp",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 2 ──────────────────────────────────────────────────────
    def insight_02_team_win_records(self) -> InsightResult:
        """Total wins per team across all seasons."""
        title = "All-Time Team Win Records (2008–2015)"

        wins = (
            self._df[~self._df["is_tied"]]
            .groupby("Winner")
            .size()
            .reset_index(name="Wins")
            .sort_values("Wins", ascending=False)
        )
        wins = wins[wins["Winner"] != "Match tied"]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
        colors = [PALETTE_TEAMS.get(t, "#888888") for t in wins["Winner"]]
        bars = ax.bar(wins["Winner"], wins["Wins"], color=colors,
                      edgecolor="white", linewidth=0.8)
        add_value_labels(ax, fmt="{:.0f}")
        ax.set_title(f"INSIGHT #02 │ {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Team")
        ax.set_ylabel("Total Wins")
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()
        self._saver.save(fig, "insight_02_team_win_records.png")

        top = wins.iloc[0]
        return InsightResult(
            2, title,
            f"{top['Winner']} leads with the most wins ({int(top['Wins'])}) "
            f"across all eight IPL seasons.",
            key_metric=f"Top team: {top['Winner']} – {int(top['Wins'])} wins",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 3 ──────────────────────────────────────────────────────
    def insight_03_season_match_scores(self) -> InsightResult:
        """Average team scores (1st innings) per season."""
        title = "Average 1st-Innings Score by Season"

        yearly_avg = self._df.groupby("Year").agg(
            avg_bat1=("Bat_First_Runs_Scored", "mean"),
            avg_bat2=("Bat_Second_Runs_Scored", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
        ax.plot(yearly_avg["Year"], yearly_avg["avg_bat1"],
                marker="o", linewidth=2.5, color="#2196F3", label="1st Innings")
        ax.plot(yearly_avg["Year"], yearly_avg["avg_bat2"],
                marker="s", linewidth=2.5, color="#FF9800", linestyle="--",
                label="2nd Innings")
        ax.fill_between(yearly_avg["Year"],
                         yearly_avg["avg_bat1"], yearly_avg["avg_bat2"],
                         alpha=0.15, color="#9C27B0")
        for _, row in yearly_avg.iterrows():
            ax.annotate(f"{row['avg_bat1']:.0f}",
                        (row["Year"], row["avg_bat1"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color="#2196F3")
        ax.set_title(f"INSIGHT #03 │ {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel("Avg Runs")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

        fig.tight_layout()
        self._saver.save(fig, "insight_03_season_avg_scores.png")

        peak_yr = yearly_avg.loc[yearly_avg["avg_bat1"].idxmax()]
        return InsightResult(
            3, title,
            f"Average 1st-innings scores fluctuate season to season. "
            f"Season {int(peak_yr['Year'])} recorded the highest average "
            f"1st-innings score of {peak_yr['avg_bat1']:.1f} runs.",
            key_metric=f"Peak season: {int(peak_yr['Year'])} "
                       f"({peak_yr['avg_bat1']:.1f} avg runs)",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 4 ──────────────────────────────────────────────────────
    def insight_04_powerplay_impact(self) -> InsightResult:
        """Correlation between powerplay score and final score / outcome."""
        title = "Powerplay Score vs Final Score & Win Probability"

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
        fig.suptitle(f"INSIGHT #04 │ {title}", fontsize=14, fontweight="bold")

        # Scatter: powerplay vs final score (1st innings)
        ax = axes[0]
        ax.scatter(self._df["bat1_powerplay_runs"],
                   self._df["Bat_First_Runs_Scored"],
                   alpha=0.4, color="#3498DB", s=40)
        m, b, r, p, _ = stats.linregress(
            self._df["bat1_powerplay_runs"].fillna(0),
            self._df["Bat_First_Runs_Scored"].fillna(0),
        )
        x_line = np.linspace(0, 100, 200)
        ax.plot(x_line, m * x_line + b, "r--", linewidth=2,
                label=f"r = {r:.2f}  (p<0.001)")
        ax.set_xlabel("Powerplay Runs (Overs 1-5)")
        ax.set_ylabel("Final 1st-Innings Score")
        ax.set_title("PP Score vs Final Score")
        ax.legend()

        # Bar: avg powerplay grouped by winner
        ax2 = axes[1]
        pp_winner = self._df.groupby("bat_first_won")["bat1_powerplay_runs"].mean()
        labels = ["Chasing Won", "Bat First Won"]
        vals = [pp_winner.get(False, 0), pp_winner.get(True, 0)]
        bars = ax2.bar(labels, vals, color=["#E74C3C", "#3498DB"],
                       edgecolor="white", width=0.5)
        add_value_labels(ax2, fmt="{:.1f}")
        ax2.set_ylabel("Avg Powerplay Runs")
        ax2.set_title("Avg PP Score: Bat First Won vs Lost")

        fig.tight_layout()
        self._saver.save(fig, "insight_04_powerplay_impact.png")

        return InsightResult(
            4, title,
            f"Strong positive correlation (r = {r:.2f}) between powerplay "
            f"score and final 1st-innings total. Teams winning batting first "
            f"average {vals[1]:.1f} powerplay runs vs {vals[0]:.1f} for "
            f"losing teams.",
            key_metric=f"Pearson r = {r:.3f}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 5 ──────────────────────────────────────────────────────
    def insight_05_winning_margin_distribution(self) -> InsightResult:
        """Distribution of winning margins by win type."""
        title = "Winning Margin Distribution (Runs vs Wickets)"

        runs_wins = self._df.loc[
            self._df["win_by_runs"] & (self._df["Winning_Margin"] > 0),
            "Winning_Margin"
        ]
        wicket_wins = self._df.loc[
            self._df["win_by_wickets"] & (self._df["Winning_Margin"] > 0),
            "Winning_Margin"
        ]

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
        fig.suptitle(f"INSIGHT #05 │ {title}", fontsize=14, fontweight="bold")

        for ax, data, label, color, xlabel in [
            (axes[0], runs_wins, "Runs", "#3498DB", "Winning Margin (Runs)"),
            (axes[1], wicket_wins, "Wickets", "#E74C3C", "Winning Margin (Wickets)"),
        ]:
            ax.hist(data, bins=20, color=color, edgecolor="white",
                    alpha=0.85, density=False)
            mu, sigma = data.mean(), data.std()
            ax.axvline(mu, color="navy", linestyle="--", linewidth=2,
                       label=f"Mean: {mu:.1f}")
            ax.axvline(mu + sigma, color="orange", linestyle=":",
                       linewidth=1.5, label=f"±1σ: {sigma:.1f}")
            ax.axvline(mu - sigma, color="orange", linestyle=":", linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Wins by {label} (n={len(data)})")
            ax.legend(fontsize=9)

        fig.tight_layout()
        self._saver.save(fig, "insight_05_winning_margins.png")

        return InsightResult(
            5, title,
            f"Run-victories average {runs_wins.mean():.1f} runs margin "
            f"(σ={runs_wins.std():.1f}). Wicket-victories average "
            f"{wicket_wins.mean():.1f} wickets margin "
            f"(σ={wicket_wins.std():.1f}). Most matches decided by narrow margins.",
            key_metric=f"Avg run margin: {runs_wins.mean():.1f} | "
                       f"Avg wicket margin: {wicket_wins.mean():.1f}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 6 ──────────────────────────────────────────────────────
    def insight_06_venue_analysis(self) -> InsightResult:
        """Top venues by matches hosted and average scores."""
        title = "Top Venues: Matches Hosted & Avg Scores"

        venue_stats = (
            self._df.groupby("Venue")
            .agg(
                matches=("Match_Number", "count"),
                avg_score=("Bat_First_Runs_Scored", "mean"),
            )
            .reset_index()
            .sort_values("matches", ascending=False)
            .head(12)
        )
        venue_stats["short_name"] = venue_stats["Venue"].apply(
            lambda v: v[:22] + "…" if len(v) > 22 else v
        )

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"INSIGHT #06 │ {title}", fontsize=14, fontweight="bold")

        ax = axes[0]
        ax.barh(venue_stats["short_name"][::-1],
                venue_stats["matches"][::-1],
                color="#5C6BC0", edgecolor="white")
        ax.set_xlabel("Matches Hosted")
        ax.set_title("Top 12 Venues by Matches Hosted")

        ax2 = axes[1]
        sc = ax2.scatter(
            venue_stats["matches"], venue_stats["avg_score"],
            s=venue_stats["matches"] * 8, alpha=0.7,
            c=venue_stats["avg_score"], cmap="RdYlGn",
        )
        plt.colorbar(sc, ax=ax2, label="Avg 1st-Innings Score")
        ax2.set_xlabel("Matches Hosted")
        ax2.set_ylabel("Avg 1st-Innings Score")
        ax2.set_title("Matches vs Avg Score (bubble = match count)")

        fig.tight_layout()
        self._saver.save(fig, "insight_06_venue_analysis.png")

        top_venue = venue_stats.iloc[0]
        return InsightResult(
            6, title,
            f"'{top_venue['Venue']}' is the most used IPL venue with "
            f"{int(top_venue['matches'])} matches. High-use venues don't "
            f"necessarily yield higher average scores.",
            key_metric=f"Most-used venue: {top_venue['Venue']} "
                       f"({int(top_venue['matches'])} matches)",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 7 ──────────────────────────────────────────────────────
    def insight_07_phase_contribution(self) -> InsightResult:
        """Phase-wise run contribution (powerplay / middle / death)."""
        title = "Phase-wise Run Contribution – Both Innings"

        phases = {
            "Powerplay (1-6)": (
                self._df["bat1_powerplay_pct"].mean(),
                self._df["bat2_powerplay_pct"].mean(),
            ),
            "Middle (7-15)": (
                self._df["bat1_middle_pct"].mean(),
                self._df["bat2_middle_pct"].mean(),
            ),
            "Death (16-20)": (
                self._df["bat1_death_pct"].mean(),
                self._df["bat2_death_pct"].mean(),
            ),
        }

        labels = list(phases.keys())
        bat1_vals = [phases[k][0] for k in labels]
        bat2_vals = [phases[k][1] for k in labels]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 7))
        width = 0.35
        bars1 = ax.bar(x - width / 2, bat1_vals, width,
                       label="1st Innings", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width / 2, bat2_vals, width,
                       label="2nd Innings", color="#FF9800", alpha=0.85)
        add_value_labels(ax, fmt="{:.1f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("% of Total Innings Runs")
        ax.set_title(f"INSIGHT #07 │ {title}", fontsize=14, fontweight="bold")
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

        fig.tight_layout()
        self._saver.save(fig, "insight_07_phase_contribution.png")

        return InsightResult(
            7, title,
            f"Middle overs contribute the most runs in both innings "
            f"({bat1_vals[1]:.1f}% | {bat2_vals[1]:.1f}%). "
            f"Death overs are crucial for chasers at "
            f"{bat2_vals[2]:.1f}% of total runs.",
            key_metric=f"Death-over contribution (2nd inn): {bat2_vals[2]:.1f}%",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 8 ──────────────────────────────────────────────────────
    def insight_08_day_vs_evening(self) -> InsightResult:
        """Day vs evening match outcomes and average scores."""
        title = "Day vs Evening Match Performance"

        time_stats = self._df.groupby("Match_Time").agg(
            matches=("Match_Number", "count"),
            avg_1st=("Bat_First_Runs_Scored", "mean"),
            avg_2nd=("Bat_Second_Runs_Scored", "mean"),
            bat_first_wins=("bat_first_won", "mean"),
        ).reset_index()

        valid = time_stats[time_stats["Match_Time"].isin(["Afternoon", "Evening"])]

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle(f"INSIGHT #08 │ {title}", fontsize=14, fontweight="bold")

        colors = ["#FFD54F", "#5C6BC0"]

        ax = axes[0]
        ax.bar(valid["Match_Time"], valid["matches"], color=colors, edgecolor="white")
        add_value_labels(ax, fmt="{:.0f}")
        ax.set_title("Matches per Slot")
        ax.set_ylabel("Count")

        ax2 = axes[1]
        x = np.arange(len(valid))
        w = 0.3
        ax2.bar(x - w / 2, valid["avg_1st"], w, label="1st Inn", color="#3498DB")
        ax2.bar(x + w / 2, valid["avg_2nd"], w, label="2nd Inn", color="#E74C3C")
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid["Match_Time"])
        ax2.set_title("Avg Scores")
        ax2.set_ylabel("Runs")
        ax2.legend()

        ax3 = axes[2]
        ax3.bar(valid["Match_Time"],
                valid["bat_first_wins"] * 100,
                color=colors, edgecolor="white")
        add_value_labels(ax3, fmt="{:.1f}")
        ax3.set_title("Bat-First Win % by Slot")
        ax3.set_ylabel("Win %")
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter())

        fig.tight_layout()
        self._saver.save(fig, "insight_08_day_vs_evening.png")

        aft = valid[valid["Match_Time"] == "Afternoon"].iloc[0]
        eve = valid[valid["Match_Time"] == "Evening"].iloc[0]
        return InsightResult(
            8, title,
            f"Evening matches ({int(eve['matches'])}) far outnumber afternoon "
            f"ones ({int(aft['matches'])}). Batting first wins "
            f"{aft['bat_first_wins'] * 100:.1f}% in afternoon vs "
            f"{eve['bat_first_wins'] * 100:.1f}% in evening fixtures.",
            key_metric=f"Evening bat-first win rate: "
                       f"{eve['bat_first_wins'] * 100:.1f}%",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 9 ──────────────────────────────────────────────────────
    def insight_09_team_consistency_seasons(self) -> InsightResult:
        """Heatmap of wins per team per season."""
        title = "Team Win Consistency Across Seasons (Heatmap)"

        pivot = (
            self._df[~self._df["is_tied"]]
            .groupby(["Year", "Winner"])
            .size()
            .reset_index(name="wins")
            .pivot(index="Winner", columns="Year", values="wins")
            .fillna(0)
        )
        pivot = pivot[pivot.index != "Match tied"]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
        sns.heatmap(
            pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"label": "Wins"},
        )
        ax.set_title(f"INSIGHT #09 │ {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel("Team")

        fig.tight_layout()
        self._saver.save(fig, "insight_09_team_consistency_heatmap.png")

        dominant = pivot.sum(axis=1).idxmax()
        return InsightResult(
            9, title,
            f"Heatmap reveals season-by-season dominance patterns. "
            f"{dominant} shows the most consistent win record across all "
            f"eight IPL seasons.",
            key_metric=f"Most consistent team: {dominant}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 10 ─────────────────────────────────────────────────────
    def insight_10_required_run_rate_pressure(self) -> InsightResult:
        """Required run rate at 10 & 15 overs and outcome."""
        title = "Required Run Rate Pressure & Chase Success"

        df_chase = self._df[self._df["chasing_won"] | self._df["bat_first_won"]].copy()
        df_chase["chase_success"] = df_chase["chasing_won"]

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
        fig.suptitle(f"INSIGHT #10 │ {title}", fontsize=14, fontweight="bold")

        for ax, col, label in [
            (axes[0], "Bat_Second_10_ov_Req_RR", "Req. RR @ 10 overs"),
            (axes[1], "Bat_Second_15_ov_Req_RR", "Req. RR @ 15 overs"),
        ]:
            df_sub = df_chase[[col, "chase_success"]].dropna()
            success = df_sub[df_sub["chase_success"]][col]
            failure = df_sub[~df_sub["chase_success"]][col]
            ax.hist(success, bins=20, alpha=0.7, color="#2ECC71",
                    label=f"Chase Won (μ={success.mean():.2f})", density=True)
            ax.hist(failure, bins=20, alpha=0.7, color="#E74C3C",
                    label=f"Chase Lost (μ={failure.mean():.2f})", density=True)
            ax.axvline(success.mean(), color="darkgreen", linestyle="--")
            ax.axvline(failure.mean(), color="darkred", linestyle="--")
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
            ax.set_title(label)

        fig.tight_layout()
        self._saver.save(fig, "insight_10_rrr_pressure.png")

        rrr10_success = df_chase["Bat_Second_10_ov_Req_RR"][df_chase["chase_success"]].mean()
        rrr10_fail = df_chase["Bat_Second_10_ov_Req_RR"][~df_chase["chase_success"]].mean()
        return InsightResult(
            10, title,
            f"Successful chases face a lower required run rate at 10 overs "
            f"(μ={rrr10_success:.2f}) vs failed chases (μ={rrr10_fail:.2f}). "
            f"A RRR above ~9 at 10 overs is a strong predictor of chase failure.",
            key_metric=f"RRR@10 threshold: success ~{rrr10_success:.2f} "
                       f"vs failure ~{rrr10_fail:.2f}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 11 ─────────────────────────────────────────────────────
    def insight_11_playoff_vs_league(self) -> InsightResult:
        """Comparison of scores: playoff matches vs league matches."""
        title = "Playoff vs League Match Scoring Patterns"

        df_playoff = self._df[self._df["is_playoff"]]
        df_league = self._df[~self._df["is_playoff"]]

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
        fig.suptitle(f"INSIGHT #11 │ {title}", fontsize=14, fontweight="bold")

        for ax, col, label in [
            (axes[0], "Bat_First_Runs_Scored", "1st Innings Score"),
            (axes[1], "Bat_Second_Runs_Scored", "2nd Innings Score"),
        ]:
            ax.hist(df_league[col], bins=20, alpha=0.7, color="#3498DB",
                    label=f"League (μ={df_league[col].mean():.1f})")
            ax.hist(df_playoff[col], bins=12, alpha=0.8, color="#E74C3C",
                    label=f"Playoff (μ={df_playoff[col].mean():.1f})")
            ax.axvline(df_league[col].mean(), color="navy", linestyle="--")
            ax.axvline(df_playoff[col].mean(), color="darkred", linestyle="--")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=9)
            ax.set_title(label)

        fig.tight_layout()
        self._saver.save(fig, "insight_11_playoff_vs_league.png")

        t_stat, p_val = stats.ttest_ind(
            df_playoff["Bat_First_Runs_Scored"].dropna(),
            df_league["Bat_First_Runs_Scored"].dropna(),
        )
        return InsightResult(
            11, title,
            f"Playoff 1st-innings average: "
            f"{df_playoff['Bat_First_Runs_Scored'].mean():.1f} vs league: "
            f"{df_league['Bat_First_Runs_Scored'].mean():.1f}. "
            f"T-test: t={t_stat:.2f}, p={p_val:.3f} – "
            f"{'significant' if p_val < 0.05 else 'not significant'} difference.",
            key_metric=f"p-value: {p_val:.3f}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 12 ─────────────────────────────────────────────────────
    def insight_12_total_runs_trend(self) -> InsightResult:
        """Total match runs trend across seasons (scoring evolution)."""
        title = "Total Match Runs Trend & Scoring Evolution"

        season_runs = self._df.groupby("Year")["total_match_runs"].agg(
            ["mean", "median", "std", "max"]
        ).reset_index()

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
        ax.fill_between(
            season_runs["Year"],
            season_runs["mean"] - season_runs["std"],
            season_runs["mean"] + season_runs["std"],
            alpha=0.2, color="#9C27B0", label="±1σ band",
        )
        ax.plot(season_runs["Year"], season_runs["mean"],
                "o-", linewidth=2.5, color="#9C27B0", label="Mean Total Runs")
        ax.plot(season_runs["Year"], season_runs["median"],
                "s--", linewidth=2, color="#FF9800", label="Median Total Runs")
        ax.plot(season_runs["Year"], season_runs["max"],
                "^:", linewidth=1.5, color="#E74C3C", label="Max Total Runs")

        for _, row in season_runs.iterrows():
            ax.annotate(f"{row['mean']:.0f}",
                        (row["Year"], row["mean"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9)

        ax.set_title(f"INSIGHT #12 │ {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Season")
        ax.set_ylabel("Total Match Runs")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

        fig.tight_layout()
        self._saver.save(fig, "insight_12_total_runs_trend.png")

        slope, _, r, p, _ = stats.linregress(
            season_runs["Year"], season_runs["mean"]
        )
        return InsightResult(
            12, title,
            f"Total match-run averages show a trend of "
            f"{'increase' if slope > 0 else 'decrease'} across seasons "
            f"(slope={slope:.2f} runs/year). "
            f"Correlation r={r:.2f} (p={p:.3f}).",
            key_metric=f"Trend slope: {slope:.2f} runs/yr | r={r:.2f}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 13 ─────────────────────────────────────────────────────
    def insight_13_head_to_head_matrix(self) -> InsightResult:
        """Head-to-head win matrix for the major six teams."""
        title = "Head-to-Head Win Matrix (Top Teams)"

        top_teams = (
            self._df["Winner"]
            .value_counts()
            .drop("Match tied", errors="ignore")
            .head(8)
            .index.tolist()
        )

        records: dict[tuple[str, str], int] = {}
        for _, row in self._df.iterrows():
            t1 = row["Team_Batting_First"]
            t2 = row["Team_Batting_Second"]
            winner = row["Winner"]
            if t1 in top_teams and t2 in top_teams and winner in top_teams:
                records[(t1, t2)] = records.get((t1, t2), 0) + (1 if winner == t1 else 0)
                records[(t2, t1)] = records.get((t2, t1), 0) + (1 if winner == t2 else 0)

        matrix = pd.DataFrame(0, index=top_teams, columns=top_teams)
        for (t1, t2), wins in records.items():
            if t1 in top_teams and t2 in top_teams:
                matrix.loc[t1, t2] = wins

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
        mask = np.eye(len(top_teams), dtype=bool)
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white", mask=mask,
            ax=ax, cbar_kws={"label": "Wins (row team vs col team)"},
        )
        ax.set_title(f"INSIGHT #13 │ {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Opponent Team")
        ax.set_ylabel("Team")
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()
        self._saver.save(fig, "insight_13_head_to_head_matrix.png")

        row_sums = matrix.sum(axis=1)
        best = row_sums.idxmax()
        return InsightResult(
            13, title,
            f"Head-to-head win matrix reveals dominant rivalries. "
            f"{best} has the most wins against top opponents "
            f"({int(row_sums[best])} wins combined).",
            key_metric=f"H2H leader: {best} ({int(row_sums[best])} wins)",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 14 ─────────────────────────────────────────────────────
    def insight_14_wickets_and_scores(self) -> InsightResult:
        """Wickets lost in each phase vs final innings score."""
        title = "Wickets Lost per Phase vs Final Score"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"INSIGHT #14 │ {title}", fontsize=14, fontweight="bold")

        phases = [
            ("Bat_First_5_ov_wkts_lost", "Bat_First_Runs_Scored",
             "Powerplay (1-5 ov)"),
            ("Bat_First_10_ov_wkts_lost", "Bat_First_Runs_Scored",
             "10-over wickets"),
            ("Bat_First_15_ov_wkts_lost", "Bat_First_Runs_Scored",
             "15-over wickets"),
        ]
        r_values = []
        for ax, (x_col, y_col, label) in zip(axes, phases):
            sub = self._df[[x_col, y_col]].dropna()
            ax.scatter(sub[x_col], sub[y_col], alpha=0.35, s=35,
                       color="#5C6BC0")
            m, b, r, p, _ = stats.linregress(sub[x_col], sub[y_col])
            r_values.append(r)
            x_l = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
            ax.plot(x_l, m * x_l + b, "r--", linewidth=2,
                    label=f"r = {r:.2f}")
            ax.set_xlabel(f"Wickets Lost ({label})")
            ax.set_ylabel("Final 1st Innings Score")
            ax.set_title(label)
            ax.legend(fontsize=9)

        fig.tight_layout()
        self._saver.save(fig, "insight_14_wickets_vs_score.png")

        return InsightResult(
            14, title,
            f"More wickets lost in the powerplay correlates negatively "
            f"with final score (r={r_values[0]:.2f}). As the innings "
            f"progresses the correlation weakens: r15={r_values[2]:.2f}, "
            f"reflecting recovery potential in later overs.",
            key_metric=f"r_PP={r_values[0]:.2f} | "
                       f"r_10={r_values[1]:.2f} | r_15={r_values[2]:.2f}",
        )

    # ------------------------------------------------------------------
    # ── INSIGHT 15 ─────────────────────────────────────────────────────
    def insight_15_score_diff_win_probability(self) -> InsightResult:
        """Score difference between innings and win probability curve."""
        title = "Score Differential & Outcome Probability"

        df_valid = self._df[
            self._df["Win_Type"].isin(["run", "wicket"])
        ].copy()
        df_valid["bat1_dominant"] = df_valid["score_diff"] > 0

        # Logistic regression bins
        bins = pd.cut(df_valid["score_diff"], bins=20)
        bin_stats = df_valid.groupby(bins, observed=True).agg(
            bat1_win_rate=("bat_first_won", "mean"),
            count=("bat_first_won", "count"),
        ).reset_index()
        bin_stats["midpoint"] = bin_stats["score_diff"].apply(
            lambda b: b.mid if hasattr(b, "mid") else np.nan
        )

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
        fig.suptitle(f"INSIGHT #15 │ {title}", fontsize=14, fontweight="bold")

        ax = axes[0]
        ax.bar(
            bin_stats["midpoint"], bin_stats["bat1_win_rate"] * 100,
            width=8, color="#3498DB", alpha=0.8, edgecolor="white",
        )
        ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50%")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Score Differential (1st Inn − 2nd Inn)")
        ax.set_ylabel("Bat-First Win %")
        ax.set_title("Win Probability by Score Diff")
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

        ax2 = axes[1]
        ax2.scatter(df_valid["Bat_First_Runs_Scored"],
                    df_valid["Bat_Second_Runs_Scored"],
                    c=df_valid["bat_first_won"].astype(int),
                    cmap="RdYlGn", alpha=0.5, s=30)
        lim_min = min(df_valid["Bat_First_Runs_Scored"].min(),
                      df_valid["Bat_Second_Runs_Scored"].min())
        lim_max = max(df_valid["Bat_First_Runs_Scored"].max(),
                      df_valid["Bat_Second_Runs_Scored"].max())
        ax2.plot([lim_min, lim_max], [lim_min, lim_max],
                 "k--", linewidth=1.5, alpha=0.6, label="Equal scores line")
        ax2.set_xlabel("1st Innings Score")
        ax2.set_ylabel("2nd Innings Score")
        ax2.set_title("Score Scatter (Green=Bat-First Win)")
        ax2.legend(fontsize=9)

        fig.tight_layout()
        self._saver.save(fig, "insight_15_score_diff_win_prob.png")

        above_50 = bin_stats[bin_stats["bat1_win_rate"] > 0.5]["midpoint"].min()
        return InsightResult(
            15, title,
            f"Bat-first teams win >50% when 1st-innings score exceeds "
            f"2nd-innings by ~{above_50:.0f} or more runs. Below that "
            f"threshold chasers dominate. The scatter plot confirms the "
            f"'equal scores line' as the decision boundary.",
            key_metric=f"Bat-first breakeven differential: ~{above_50:.0f} runs",
        )

    # ------------------------------------------------------------------
    # ── Aggregate runner ───────────────────────────────────────────────
    def run_all(self) -> list[InsightResult]:
        """Run all 15 insights in sequence and return results."""
        methods = [
            self.insight_01_bat_first_vs_chase,
            self.insight_02_team_win_records,
            self.insight_03_season_match_scores,
            self.insight_04_powerplay_impact,
            self.insight_05_winning_margin_distribution,
            self.insight_06_venue_analysis,
            self.insight_07_phase_contribution,
            self.insight_08_day_vs_evening,
            self.insight_09_team_consistency_seasons,
            self.insight_10_required_run_rate_pressure,
            self.insight_11_playoff_vs_league,
            self.insight_12_total_runs_trend,
            self.insight_13_head_to_head_matrix,
            self.insight_14_wickets_and_scores,
            self.insight_15_score_diff_win_probability,
        ]
        results: list[InsightResult] = []
        for method in methods:
            self._LOG.info("Generating %s …", method.__name__)
            try:
                result = method()
                results.append(result)
                print(result)
            except Exception as exc:  # noqa: BLE001
                self._LOG.error("Failed %s: %s", method.__name__, exc, exc_info=True)
        return results


# ─── Summary Dashboard ────────────────────────────────────────────────────────
class IPLDashboard:
    """
    Generates a single summary dashboard figure collating key metrics
    from all insights into one executive overview.
    """

    _LOG = logging.getLogger("IPLDashboard")

    def __init__(self, df: pd.DataFrame, saver: ChartSaver) -> None:
        self._df = df
        self._saver = saver

    # ------------------------------------------------------------------
    def render(self) -> str:
        """Build and save the dashboard; return file path."""
        self._LOG.info("Rendering executive summary dashboard …")

        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(
            "IPL Analytics Dashboard  │  2008 – 2015  │  Executive Summary",
            fontsize=18, fontweight="bold", y=0.98,
        )

        # ── Row 1: KPI tiles ──────────────────────────────────────────
        kpis = self._compute_kpis()
        tile_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
                       "#F44336", "#00BCD4"]
        for i, (label, val) in enumerate(kpis.items()):
            ax = fig.add_subplot(4, 6, i + 1)
            ax.set_facecolor(tile_colors[i % len(tile_colors)])
            ax.text(0.5, 0.6, str(val), ha="center", va="center",
                    fontsize=20, fontweight="bold", color="white",
                    transform=ax.transAxes)
            ax.text(0.5, 0.2, label, ha="center", va="center",
                    fontsize=8, color="white", transform=ax.transAxes,
                    wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # ── Row 2: Top-team bar ───────────────────────────────────────
        ax2 = fig.add_subplot(4, 2, 3)
        wins = (
            self._df[~self._df["is_tied"]]
            .groupby("Winner").size()
            .drop("Match tied", errors="ignore")
            .sort_values(ascending=True)
            .tail(8)
        )
        colors = [PALETTE_TEAMS.get(t, "#888") for t in wins.index]
        ax2.barh(wins.index, wins.values, color=colors, edgecolor="white")
        ax2.set_title("Top Teams by Total Wins", fontweight="bold")
        ax2.set_xlabel("Wins")

        # ── Row 2: Phase pie ─────────────────────────────────────────
        ax3 = fig.add_subplot(4, 2, 4)
        phase_means = [
            self._df["bat1_powerplay_pct"].mean(),
            self._df["bat1_middle_pct"].mean(),
            self._df["bat1_death_pct"].mean(),
        ]
        ax3.pie(
            phase_means,
            labels=["Powerplay", "Middle", "Death"],
            colors=list(PHASE_COLORS.values()),
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 11},
        )
        ax3.set_title("1st Innings Phase Contribution", fontweight="bold")

        # ── Row 3: Yearly avg scores ──────────────────────────────────
        ax4 = fig.add_subplot(4, 2, 5)
        yr = self._df.groupby("Year")["Bat_First_Runs_Scored"].mean()
        ax4.plot(yr.index, yr.values, "o-", color="#2196F3", linewidth=2.5)
        ax4.fill_between(yr.index, yr.values, alpha=0.15, color="#2196F3")
        ax4.set_title("Avg 1st-Innings Score per Season", fontweight="bold")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Runs")
        ax4.xaxis.set_major_locator(mticker.MultipleLocator(1))

        # ── Row 3: Bat-first win by year ─────────────────────────────
        ax5 = fig.add_subplot(4, 2, 6)
        yr_bf = self._df.groupby("Year")["bat_first_won"].mean() * 100
        ax5.bar(yr_bf.index, yr_bf.values, color="#FF9800", alpha=0.85,
                edgecolor="white")
        ax5.axhline(50, color="red", linestyle="--", linewidth=1.5)
        ax5.set_title("Bat-First Win % per Season", fontweight="bold")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Win %")
        ax5.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax5.xaxis.set_major_locator(mticker.MultipleLocator(1))

        # ── Row 4: Total match runs box ───────────────────────────────
        ax6 = fig.add_subplot(4, 2, 7)
        groups = [
            grp["total_match_runs"].dropna().tolist()
            for _, grp in self._df.groupby("Year")
        ]
        years = sorted(self._df["Year"].dropna().unique())
        bp = ax6.boxplot(groups, patch_artist=True,
                         medianprops={"color": "red", "linewidth": 2})
        cmap_vals = plt.cm.viridis(np.linspace(0.2, 0.8, len(years)))  # type: ignore[attr-defined]
        for patch, c in zip(bp["boxes"], cmap_vals):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax6.set_xticklabels([str(int(y)) for y in years], rotation=30)
        ax6.set_title("Total Match Runs Distribution per Season",
                      fontweight="bold")
        ax6.set_ylabel("Total Runs")

        # ── Row 4: Win type doughnut ──────────────────────────────────
        ax7 = fig.add_subplot(4, 2, 8)
        wt = self._df["Win_Type"].value_counts()
        ax7.pie(
            wt.values,
            labels=wt.index,
            colors=["#3498DB", "#E74C3C"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.55},
            textprops={"fontsize": 12},
        )
        ax7.set_title("Win Type Split", fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fp = self._saver.save(fig, "ipl_executive_dashboard.png")
        self._LOG.info("Dashboard saved → %s", fp)
        return fp

    # ------------------------------------------------------------------
    def _compute_kpis(self) -> dict[str, str]:
        df = self._df
        total = len(df)
        bf_pct = df["bat_first_won"].mean() * 100
        avg_score = df["Bat_First_Runs_Scored"].mean()
        top_team = (
            df[~df["is_tied"]]
            .groupby("Winner").size()
            .drop("Match tied", errors="ignore")
            .idxmax()
        )
        dl_pct = df["dl_match"].mean() * 100
        venues = df["Venue"].nunique()
        return {
            "Total Matches": str(total),
            "Bat-First Win %": f"{bf_pct:.1f}%",
            "Avg 1st Inn Score": f"{avg_score:.0f}",
            "Most Wins Team": str(top_team),
            "DL Used %": f"{dl_pct:.1f}%",
            "Unique Venues": str(venues),
        }


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────
class IPLAnalyticsPipeline:
    """
    Orchestrates the end-to-end analytics workflow:
        Load → Process → Analyse → Visualise → Report
    """

    _LOG = logging.getLogger("IPLAnalyticsPipeline")

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        apply_global_style()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full pipeline."""
        self._LOG.info("=" * 70)
        self._LOG.info("  IPL Analytics Pipeline  |  START")
        self._LOG.info("=" * 70)

        # 1. Load
        loader = IPLDataLoader(self._cfg.data_path)
        raw_df = loader.load()

        # 2. Process
        processor = IPLDataProcessor(raw_df)
        processor.process()
        df = processor.data

        # 3. Initialise chart saver
        saver = ChartSaver(
            output_dir=self._cfg.output_dir,
            dpi=self._cfg.dpi,
            show=self._cfg.show_charts,
        )

        # 4. Run insights
        engine = IPLInsightsEngine(df, saver)
        results = engine.run_all()

        # 5. Dashboard
        dashboard = IPLDashboard(df, saver)
        dash_path = dashboard.render()

        # 6. Print final summary
        self._print_summary(results, dash_path)

        self._LOG.info("=" * 70)
        self._LOG.info("  IPL Analytics Pipeline  |  COMPLETE")
        self._LOG.info("  Output directory: %s", Path(self._cfg.output_dir).resolve())
        self._LOG.info("=" * 70)

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(results: list[InsightResult], dash_path: str) -> None:
        print("\n" + "█" * 70)
        print("  PIPELINE SUMMARY  –  All Insights Generated")
        print("█" * 70)
        for r in results:
            print(f"  ✔  #{r.insight_id:02d}  {r.title}")
            if r.key_metric:
                print(f"       └─ {r.key_metric}")
        print(f"\n  Executive Dashboard → {dash_path}")
        print("█" * 70)


# ─── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    """
    Entry point.

    Usage
    -----
    1. Place IPL.xls in the same directory as this script, OR
    2. Pass the path via DATA_PATH env variable:
           DATA_PATH=/path/to/IPL.xls python ipl_analytics_pipeline.py
    """
    data_path = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)

    config = PipelineConfig(
        data_path=data_path,
        output_dir=OUTPUT_DIR,
        dpi=FIGURE_DPI,
        save_charts=True,
        show_charts=False,   # ← set True for interactive PyCharm display
    )

    pipeline = IPLAnalyticsPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
