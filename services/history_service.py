import os
import json

class HistoryService:
    def __init__(self, history_file="history/history.json"):
        self.history_file = history_file
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        if not os.path.exists(history_file):
            with open(history_file, "w") as f:
                json.dump([], f)

    def load_history(self):
        if not os.path.exists(self.history_file):
            return []

        try:
            with open(self.history_file, "r") as f:
                data = f.read().strip()
                if not data:
                    return []
                return json.loads(data)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def save_record(self, record: dict):
        history = self.load_history()
        history.append(record)
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=4)

    def clear_history(self):
        with open(self.history_file, "w") as f:
            json.dump([], f, indent=4)

    def clear_history_and_files(self):
        history = self.load_history()

        for record in history:
            original = record.get("original_filename")
            processed = record.get("processed_filename")

            if original and os.path.exists(original):
                try:
                    os.remove(original)
                except Exception as e:
                    print(f"Не удалось удалить оригинальный файл {original}: {e}")

            if processed and os.path.exists(processed):
                try:
                    os.remove(processed)
                except Exception as e:
                    print(f"Не удалось удалить обработанный файл {processed}: {e}")

        self.clear_history()
