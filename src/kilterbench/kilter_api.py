import io
import time
from typing import Any, Literal, get_args
import uuid
import sqlite3
import zipfile

import pandas as pd
import requests

from .types import ClimbStats

TableLiteral = Literal[
    "products",
    "product_sizes",
    "holes",
    "leds",
    "products_angles",
    "layouts",
    "product_sizes_layouts_sets",
    "placements",
    "sets",
    "placement_roles",
    "climbs",
    "climb_stats",
    "beta_links",
    "attempts",
    "kits",
    "users",
    "walls",
    "draft_climbs",
    "ascents",
    "bids",
    "tags",
    "circuits",
]
_ALL_TABLES = list(get_args(TableLiteral))
_INDEX_COLS = {
    "climbs": ["uuid_upper"],
    "climb_stats": ["climb_uuid_upper", "angle"],
    "circuits": ["uuid_upper"],
}


class KilterAPI:
    _URL: str = "https://kilterboardapp.com"
    _sync_times: dict[TableLiteral, str]

    token: str
    user_id: int
    tables: dict[TableLiteral, pd.DataFrame]
    difficulty_grades: pd.DataFrame

    def __init__(self, username: str, password: str) -> None:
        self.reset()
        response = requests.post(
            f"{self._URL}/sessions",
            data={
                "username": username,
                "password": password,
                "tou": "accepted",
                "pp": "accepted",
                "ua": "app",
            },
        )
        response.raise_for_status()
        payload = response.json()
        self.token = payload["session"]["token"]
        self.user_id = payload["session"]["user_id"]
        self._download_db()
        self.sync()

    def sync(self, tables: TableLiteral | list[TableLiteral] | None = None) -> None:
        tables = tables or _ALL_TABLES
        if isinstance(tables, str):
            tables = [tables]
        # Set the sync time to be some time in the future for all tables that aren't being requested
        sync_times = self._sync_times.copy()
        sync_times.update(
            {
                table: time.strftime(
                    "%Y-%m-%d %H:%M:%S.000000", time.gmtime(time.time() + 1000)
                )
                for table in _ALL_TABLES
                if table not in tables
            }
        )
        complete = False
        while not complete:
            response = requests.post(
                f"{self._URL}/sync",
                data=sync_times,
                cookies={"token": self.token},
            )
            response.raise_for_status()
            json = response.json()
            updates = json.get("user_syncs", []) + json.get("shared_syncs", [])
            for s in updates:
                table_name = s["table_name"]
                synced_at = s["last_synchronized_at"]
                sync_times[table_name] = synced_at
                self._sync_times[table_name] = synced_at
            complete = json["_complete"]
            for table_name, response_data in json.items():
                if table_name in tables:
                    self._update_table(table_name, response_data)

    def _update_table(
        self, table_name: TableLiteral, data: list[dict[str, Any]]
    ) -> None:
        cols = list(data[0].keys())
        df = pd.DataFrame.from_dict(
            {i: d for i, d in enumerate(data)}, orient="index", columns=cols
        )
        # Ensure any uuid columns are all uppercase
        uuid_cols = [c for c in cols if c.endswith("uuid")]
        for col in uuid_cols:
            df[f"{col}_upper"] = df[col].str.upper()
        if table_name in _INDEX_COLS:
            # Set the index to the index col(s), update any existing rows,
            # then add any that arent in the index
            table = self.tables[table_name].set_index(_INDEX_COLS[table_name])
            df = df.set_index(_INDEX_COLS[table_name])
            # Update the table, except for any UUID columns. For some reason some climb
            # uuids change case between updates and we want to maintain the original case
            table.update(df.drop(uuid_cols, axis=1))
            df = pd.concat(
                [table, df.loc[df.index.difference(table.index)]],
                verify_integrity=True,
            ).reset_index()
        elif table_name in self.tables:
            table = self.tables[table_name]
            df = pd.concat([table, df], ignore_index=True)
        self.tables[table_name] = df

    def get_climb_stats(self, climb_uuid: str, angle: int) -> ClimbStats:
        response = requests.get(
            f"{self._URL}/climbs/{climb_uuid}/info",
            params={"angle": angle},
            cookies={"token": self.token},
        )
        response.raise_for_status()
        return response.json()

    def make_new_circuit(
        self,
        name: str,
        description: str = "",
        colour: Literal[
            "red", "orange", "green", "blue", "purple", "pink", "grey"
        ] = "red",
        is_public: bool = False,
    ) -> str:
        # Check if a circuit with this name exists already
        self.sync("circuits")
        for _, row in self.tables["circuits"].iterrows():
            if row["name"] == name:
                return row["uuid"]
        COLOURS = {
            "red": "FF0000",
            "orange": "FF8000",
            "green": "00CC00",
            "blue": "0000FF",
            "purple": "8000FF",
            "pink": "FF00FF",
            "grey": "808080",
        }
        circuit_id = str(uuid.uuid4()).replace("-", "")
        response = requests.post(
            f"{self._URL}/circuits/save",
            data={
                "uuid": circuit_id,
                "user_id": self.user_id,
                "name": name,
                "description": description,
                "color": COLOURS[colour],
                "is_public": 1 if is_public else 0,
            },
            cookies={"token": self.token},
        )
        response.raise_for_status()
        payload = response.json()
        return circuit_id

    def set_circuit(self, circuit_id: str, climb_ids: list[str]) -> None:
        self.sync("circuits")
        # Get current circuits that this climb is part of
        response = requests.post(
            f"{self._URL}/circuit_climbs/save",
            data={
                "circuit_uuid": circuit_id,
                "climb_uuids[]": climb_ids,
            },
            cookies={"token": self.token},
        )
        response.raise_for_status()
        payload = response.json()
        self.sync("circuits")

    def reset(self) -> None:
        self.tables = {}
        self._sync_times: dict[TableLiteral, str] = {
            table: "1970-01-01 00:00:00.000000" for table in _ALL_TABLES
        }

    def _download_db(self) -> None:
        """
        The sqlite3 database is stored in the assets folder of the APK files for the Android app of each board.

        This function downloads the latest APK file for the board's Android app and extracts the database from it.
        :param board: The board to download the database for.
        :param output_file: The file to write the database to.
        """
        # Based on DB Download function from BoardLib
        # https://github.com/lemeryfertitta/BoardLib/blob/7cad7040e13488b898c319cd4a2f8f3f02a026f5/src/boardlib/db/aurora.py#L21C1-L41C66
        response = requests.get(
            "https://d.apkpure.net/b/APK/com.auroraclimbing.kilterboard",
            params={"version": "latest"},
            # Some user-agent is required, 403 if not included
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            },
        )
        response.raise_for_status()
        apk_file = io.BytesIO(response.content)
        with zipfile.ZipFile(apk_file, "r") as zip_file:
            db = zip_file.read("assets/db.sqlite3")
            with sqlite3.connect(":memory:") as conn:
                conn.deserialize(db)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                db_tables = cursor.fetchall()
                self.tables = {
                    table: pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    for table in _ALL_TABLES
                    if (table,) in db_tables
                }
                # Also read some DB specific tables, these arent ones that we can sync from the server
                self.difficulty_grades = pd.read_sql_query(
                    "SELECT * FROM difficulty_grades", conn
                )
                # Update sync times
                syncs = pd.read_sql_query("SELECT * FROM shared_syncs", conn)
                self._sync_times.update({t: d for t, d in syncs.apply(tuple, axis=1)})
        # Ensure that any UUID columns are all upper case
        for table_name, table in self.tables.items():
            for col in [c for c in table.columns if c.endswith("uuid")]:
                table[f"{col}_upper"] = table[col].str.upper()
