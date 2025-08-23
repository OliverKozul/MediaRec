import csv
import os
import signal

from core.logger import Logger
from tmdb_loader import TMDbLoader
from concurrent.futures import ThreadPoolExecutor

class DatasetCreator:
    """Creates a dataset by fetching media data from TMDb."""
    def __init__(self, media_csv_path: str, links_csv_path: str, output_csv_path: str):
        self.media_csv_path = media_csv_path
        self.links_csv_path = links_csv_path
        self.output_csv_path = output_csv_path
        self.media_id_to_tmdb_id = {}
        self.enriched_media = []
        self.error_count = 0
        self.max_errors = 1000
        self.starting_id = 0
        self.tmdb_loader = TMDbLoader()
        self.logger = Logger().get_logger("DatasetCreator")
        signal.signal(signal.SIGINT, self.handle_exit)

    def load_links(self) -> None:
        with open(self.links_csv_path, "r") as links_file:
            reader = csv.DictReader(links_file)
            for row in reader:
                if "tmdb_id" in row and row["tmdb_id"].isdigit():
                    self.media_id_to_tmdb_id[int(row["media_id"])] = int(row["tmdb_id"])

    def resume_from_last_id(self) -> None:
        if os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, "r", newline="", encoding="utf-8") as output_file:
                reader = csv.DictReader(output_file)
                rows_out = list(reader)
                if rows_out:
                    self.starting_id = int(rows_out[-1]["media_id"])
                    self.logger.info(f"Resuming from media ID: {self.starting_id}")

    def fetch_tmdb_data_parallel(self, media_rows: list) -> list:
        def process_row(row):
            media_id = int(row["media_id"])
            tmdb_id = self.media_id_to_tmdb_id.get(media_id)
            if media_id % 100 == 0:
                self.logger.info(f"Processing media ID: {media_id}")

            if tmdb_id:
                tmdb_data = self.tmdb_loader.fetch_tmdb_data(tmdb_id)
                if tmdb_data:
                    return {
                        "media_id": media_id,
                        "media_type": tmdb_data["media_type"],
                        "title": row["title"],
                        "genres": row["genres"],
                        "actors": ",".join(tmdb_data["actors"]),
                        "director": tmdb_data["director"],
                        "description": tmdb_data["description"],
                        "poster_path": tmdb_data["poster_path"]
                    }
                else:
                    self.error_count += 1
                    self.logger.error(f"Failed to fetch TMDb data for media ID {media_id} (TMDb ID {tmdb_id}). Error count: {self.error_count}/{self.max_errors}")
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_row, media_rows))
        return [result for result in results if result is not None]

    def create_dataset(self) -> None:
        self.load_links()
        self.resume_from_last_id()

        with open(self.media_csv_path, "r", encoding="utf-8") as media_file:
            reader = csv.DictReader(media_file)
            rows = list(reader)
            rows_to_process = [row for row in rows if int(row["media_id"]) > self.starting_id]

            batch_size = 512
            while rows_to_process and self.error_count < self.max_errors:
                batch = rows_to_process[:batch_size]
                rows_to_process = rows_to_process[batch_size:]

                self.enriched_media.extend(self.fetch_tmdb_data_parallel(batch))

        if self.error_count >= self.max_errors:
            self.logger.error("Maximum error count reached. Dataset creation halted.")

        self.save_dataset()

    def handle_exit(self, signum: int, frame: signal.FrameType) -> None:
        if not hasattr(self, "exit_handled"):
            self.exit_handled = True
            self.logger.info("Program interrupted. Saving progress...")
            self.save_dataset()
            exit(0)

    def save_dataset(self) -> None:
        file_exists = os.path.exists(self.output_csv_path)
        with open(self.output_csv_path, "a", newline="", encoding="utf-8") as output_file:
            fieldnames = ["media_id", "media_type", "title", "genres", "actors", "director", "description", "poster_path"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.enriched_media)
        self.logger.info(f"Dataset saved to {self.output_csv_path}. Total records: {len(self.enriched_media)}")

if __name__ == "__main__":
    # dataset_folder = "ml-latest-small/"
    dataset_folder = "ml-32m/"

    creator = DatasetCreator(
        media_csv_path=dataset_folder + "movies.csv",
        links_csv_path=dataset_folder + "links.csv",
        output_csv_path=dataset_folder + "complete_media_with_imgs.csv"
    )
    creator.create_dataset()
