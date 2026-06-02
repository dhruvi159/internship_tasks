"""
====================================================================
Production Grade Airbnb Data Analytics & Visualization Pipeline
====================================================================
Author  : dhruvi lolariya
Dataset : Airbnb_data.csv

Requirements:
    pip install pandas numpy matplotlib seaborn scipy

Run:
    python airbnb_pipeline.py
====================================================================
"""

from pathlib import Path
from typing import Dict

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


class Config:
    """Application configuration."""

    DATASET_PATH = "./final_industry_task/Airbnb_data.csv"
    OUTPUT_FOLDER = "outputs"
    FIGURE_SIZE = (12, 6)
    STYLE = "darkgrid"
    DPI = 300


class DataLoader:
    """Handles dataset loading."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset."""

        try:
            dataframe = pd.read_csv(self.file_path)

            print("\nDataset loaded successfully.")
            print(f"Dataset Shape: {dataframe.shape}")

            return dataframe

        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Dataset not found at: {self.file_path}"
            ) from error


class DataInspector:
    """Handles dataset inspection."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def display_basic_information(self) -> None:
        """Display dataset information."""

        print("\n================ DATASET INFORMATION ================")
        print(self.dataframe.info())

        print("\n================ MISSING VALUES ================")
        print(self.dataframe.isnull().sum())

        print("\n================ DUPLICATE ROWS ================")
        print(self.dataframe.duplicated().sum())

        print("\n================ DESCRIPTIVE STATISTICS ================")
        print(self.dataframe.describe(include="all"))


class DataPreprocessor:
    """Handles preprocessing operations."""

    def __init__(self, dataframe: pd.DataFrame):
        self.original_dataframe = dataframe
        self.cleaned_dataframe = dataframe.copy()

    def preprocess(self) -> pd.DataFrame:
        """Execute preprocessing pipeline."""

        self._remove_duplicates()
        self._handle_missing_values()
        self._convert_data_types()
        self._remove_outliers()
        self._feature_engineering()

        print("\nPreprocessing completed successfully.")

        return self.cleaned_dataframe

    def _remove_duplicates(self) -> None:
        """Remove duplicate rows."""

        before_rows = self.cleaned_dataframe.shape[0]

        self.cleaned_dataframe.drop_duplicates(inplace=True)

        after_rows = self.cleaned_dataframe.shape[0]

        print(f"\nDuplicates Removed: {before_rows - after_rows}")

    def _handle_missing_values(self) -> None:
        """Handle missing values safely."""

        fill_columns = [
            "name",
            "host_name"
        ]

        for column in fill_columns:
            if column in self.cleaned_dataframe.columns:
                self.cleaned_dataframe[column] = (
                    self.cleaned_dataframe[column].fillna("Unknown")
                )

        if "reviews_per_month" in self.cleaned_dataframe.columns:
            self.cleaned_dataframe["reviews_per_month"] = (
                self.cleaned_dataframe["reviews_per_month"].fillna(0)
            )

        if "last_review" in self.cleaned_dataframe.columns:
            self.cleaned_dataframe["last_review"] = (
                self.cleaned_dataframe["last_review"].fillna(
                    "Not Available"
                )
            )

    def _convert_data_types(self) -> None:
        """Convert required datatypes."""

        if "last_review" in self.cleaned_dataframe.columns:
            self.cleaned_dataframe["last_review"] = pd.to_datetime(
                self.cleaned_dataframe["last_review"],
                errors="coerce"
            )

    def _remove_outliers(self) -> None:
        """Remove price outliers using IQR."""

        if "price" not in self.cleaned_dataframe.columns:
            return

        q1 = self.cleaned_dataframe["price"].quantile(0.25)
        q3 = self.cleaned_dataframe["price"].quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        self.cleaned_dataframe = self.cleaned_dataframe[
            (
                self.cleaned_dataframe["price"] >= lower_limit
            )
            &
            (
                self.cleaned_dataframe["price"] <= upper_limit
            )
        ]

        print("\nPrice outliers removed successfully.")

    def _feature_engineering(self) -> None:
        """Create engineered analytical features."""

        if "price" in self.cleaned_dataframe.columns:
            self.cleaned_dataframe["price_category"] = pd.cut(
                self.cleaned_dataframe["price"],
                bins=[0, 100, 250, 500, 1000],
                labels=[
                    "Budget",
                    "Standard",
                    "Premium",
                    "Luxury"
                ]
            )

        if (
            "number_of_reviews" in self.cleaned_dataframe.columns
            and "reviews_per_month" in self.cleaned_dataframe.columns
        ):
            self.cleaned_dataframe["review_score"] = (
                self.cleaned_dataframe["number_of_reviews"]
                *
                self.cleaned_dataframe["reviews_per_month"]
            )

        if "availability_365" in self.cleaned_dataframe.columns:
            self.cleaned_dataframe["availability_ratio"] = (
                self.cleaned_dataframe["availability_365"] / 365
            )


class StatisticalAnalyzer:
    """Performs business analytics and insight generation."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def generate_insights(self) -> Dict[str, str]:
        """Generate business insights."""

        insights = {}

        insights["Insight 1"] = (
            f"Average Airbnb price: "
            f"${self.dataframe['price'].mean():.2f}"
        )

        insights["Insight 2"] = (
            f"Median Airbnb price: "
            f"${self.dataframe['price'].median():.2f}"
        )

        expensive_group = self.dataframe.groupby(
            "neighbourhood_group"
        )["price"].mean().idxmax()

        insights["Insight 3"] = (
            f"Most expensive neighbourhood group: "
            f"{expensive_group}"
        )

        affordable_group = self.dataframe.groupby(
            "neighbourhood_group"
        )["price"].mean().idxmin()

        insights["Insight 4"] = (
            f"Most affordable neighbourhood group: "
            f"{affordable_group}"
        )

        most_common_room = (
            self.dataframe["room_type"].mode()[0]
        )

        insights["Insight 5"] = (
            f"Most common room type: "
            f"{most_common_room}"
        )

        highest_reviewed = self.dataframe.loc[
            self.dataframe["number_of_reviews"].idxmax(),
            "name"
        ]

        insights["Insight 6"] = (
            f"Highest reviewed property: "
            f"{highest_reviewed}"
        )

        top_host = self.dataframe.groupby(
            "host_name"
        )["calculated_host_listings_count"].sum().idxmax()

        insights["Insight 7"] = (
            f"Host with maximum listings: "
            f"{top_host}"
        )

        busiest_area = self.dataframe.groupby(
            "neighbourhood_group"
        )["number_of_reviews"].sum().idxmax()

        insights["Insight 8"] = (
            f"Most reviewed neighbourhood group: "
            f"{busiest_area}"
        )

        correlation, _ = pearsonr(
            self.dataframe["price"],
            self.dataframe["availability_365"]
        )

        insights["Insight 9"] = (
            f"Correlation between price and availability: "
            f"{correlation:.2f}"
        )

        expensive_room = self.dataframe.groupby(
            "room_type"
        )["price"].mean().idxmax()

        insights["Insight 10"] = (
            f"Most expensive room type: "
            f"{expensive_room}"
        )

        affordable_room = self.dataframe.groupby(
            "room_type"
        )["price"].mean().idxmin()

        insights["Insight 11"] = (
            f"Most affordable room type: "
            f"{affordable_room}"
        )

        highly_available = self.dataframe[
            self.dataframe["availability_365"] > 300
        ].shape[0]

        insights["Insight 12"] = (
            f"Listings available for more than 300 days: "
            f"{highly_available}"
        )

        luxury_count = self.dataframe[
            self.dataframe["price_category"] == "Luxury"
        ].shape[0]

        insights["Insight 13"] = (
            f"Luxury listings count: "
            f"{luxury_count}"
        )

        average_minimum_nights = (
            self.dataframe["minimum_nights"].mean()
        )

        insights["Insight 14"] = (
            f"Average minimum nights required: "
            f"{average_minimum_nights:.2f}"
        )

        highest_neighbourhood = self.dataframe.groupby(
            "neighbourhood"
        )["price"].mean().idxmax()

        insights["Insight 15"] = (
            f"Highest average price neighbourhood: "
            f"{highest_neighbourhood}"
        )

        return insights

    @staticmethod
    def print_insights(
        insights: Dict[str, str]
    ) -> None:
        """Print insights."""

        print("\n================ BUSINESS INSIGHTS ================")

        for key, value in insights.items():
            print(f"{key}: {value}")


class VisualizationEngine:
    """Handles all visualizations."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

        Path(Config.OUTPUT_FOLDER).mkdir(exist_ok=True)

        sns.set_style(Config.STYLE)

    def generate_all_visualizations(self) -> None:
        """Generate all plots."""

        self._price_distribution()
        self._room_type_distribution()
        self._price_by_neighbourhood()
        self._availability_distribution()
        self._reviews_vs_price()
        self._correlation_heatmap()
        self._top_hosts()
        self._minimum_nights()
        self._room_price_comparison()
        self._listing_count()

        print("\nAll visualizations generated successfully.")

    def _save_plot(self, filename: str) -> None:
        """Save visualization."""

        plt.tight_layout()

        plt.savefig(
            f"{Config.OUTPUT_FOLDER}/{filename}",
            dpi=Config.DPI
        )

        plt.close()

    def _price_distribution(self) -> None:
        """Generate price distribution."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.histplot(
            self.dataframe["price"],
            bins=50,
            kde=True
        )

        plt.title("Price Distribution")

        self._save_plot("price_distribution.png")

    def _room_type_distribution(self) -> None:
        """Generate room type distribution."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.countplot(
            data=self.dataframe,
            x="room_type"
        )

        plt.title("Room Type Distribution")

        self._save_plot("room_type_distribution.png")

    def _price_by_neighbourhood(self) -> None:
        """Generate neighbourhood price analysis."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        prices = self.dataframe.groupby(
            "neighbourhood_group"
        )["price"].mean().sort_values()

        prices.plot(kind="bar")

        plt.title("Average Price by Neighbourhood Group")

        self._save_plot("price_by_neighbourhood.png")

    def _availability_distribution(self) -> None:
        """Generate availability analysis."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.histplot(
            self.dataframe["availability_365"],
            bins=30,
            kde=True
        )

        plt.title("Availability Distribution")

        self._save_plot("availability_distribution.png")

    def _reviews_vs_price(self) -> None:
        """Generate reviews vs price analysis."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.scatterplot(
            data=self.dataframe,
            x="number_of_reviews",
            y="price"
        )

        plt.title("Reviews vs Price")

        self._save_plot("reviews_vs_price.png")

    def _correlation_heatmap(self) -> None:
        """Generate heatmap."""

        plt.figure(figsize=(10, 8))

        numerical_dataframe = (
            self.dataframe.select_dtypes(include=[np.number])
        )

        sns.heatmap(
            numerical_dataframe.corr(),
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        self._save_plot("correlation_heatmap.png")

    def _top_hosts(self) -> None:
        """Generate top host analysis."""

        plt.figure(figsize=(14, 6))

        top_hosts = self.dataframe.groupby(
            "host_name"
        )["calculated_host_listings_count"].sum().sort_values(
            ascending=False
        ).head(10)

        top_hosts.plot(kind="bar")

        plt.title("Top Hosts by Listing Count")

        self._save_plot("top_hosts.png")

    def _minimum_nights(self) -> None:
        """Generate minimum nights analysis."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.boxplot(
            data=self.dataframe,
            x="minimum_nights"
        )

        plt.title("Minimum Nights Distribution")

        self._save_plot("minimum_nights_distribution.png")

    def _room_price_comparison(self) -> None:
        """Generate room price comparison."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.boxplot(
            data=self.dataframe,
            x="room_type",
            y="price"
        )

        plt.title("Room Type vs Price")

        self._save_plot("room_price_comparison.png")

    def _listing_count(self) -> None:
        """Generate listing count analysis."""

        plt.figure(figsize=Config.FIGURE_SIZE)

        sns.countplot(
            data=self.dataframe,
            x="neighbourhood_group"
        )

        plt.title("Neighbourhood Listing Count")

        self._save_plot("listing_count.png")


class PipelineController:
    """Main analytics pipeline."""

    def __init__(self):
        self.dataframe = None
        self.cleaned_dataframe = None

    def execute_pipeline(self) -> None:
        """Execute complete pipeline."""

        print("\n================================================")
        print("AIRBNB INDUSTRIAL ANALYTICS PIPELINE STARTED")
        print("================================================")

        loader = DataLoader(Config.DATASET_PATH)

        self.dataframe = loader.load_dataset()

        inspector = DataInspector(self.dataframe)
        inspector.display_basic_information()

        preprocessor = DataPreprocessor(self.dataframe)

        self.cleaned_dataframe = preprocessor.preprocess()

        analyzer = StatisticalAnalyzer(
            self.cleaned_dataframe
        )

        insights = analyzer.generate_insights()

        analyzer.print_insights(insights)

        visualizer = VisualizationEngine(
            self.cleaned_dataframe
        )

        visualizer.generate_all_visualizations()

        print("\n================================================")
        print("PIPELINE EXECUTED SUCCESSFULLY")
        print("================================================")


def main() -> None:
    """Application entry point."""

    pipeline = PipelineController()

    pipeline.execute_pipeline()


if __name__ == "__main__":
    main()
