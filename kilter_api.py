from collections import defaultdict
import io
import time
from typing import TypedDict, Literal, get_args
import uuid
import zipfile

import pandas as pd
import requests
import sqlite3

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
    "climbs": ["uuid"],
    "climb_stats": ["climb_uuid", "angle"],
    "circuits": ["uuid"],
}


class GradeHistogramEntry(TypedDict):
    difficulty: int
    count: int


class QualityHistogramEntry(TypedDict):
    quality: int
    count: int


class AscentEntry(TypedDict):
    attempt_id: int
    angle: int
    quality: int
    difficulty: int
    is_benchmark: bool
    is_mirror: bool
    comment: str
    climbed_at: str
    user_id: int
    user_username: int
    user_avatar_image: None | str


class ClimbStats(TypedDict):
    difficulty: list[GradeHistogramEntry]
    quality: list[QualityHistogramEntry]
    ascents: list[AscentEntry]


class KilterAPI:
    _URL: str = "https://kilterboardapp.com"
    _sync_times: dict[TableLiteral, str] = {
        table: "1970-01-01 00:00:00.000000" for table in _ALL_TABLES
    }

    token: str
    user_id: int
    tables: dict[TableLiteral, pd.DataFrame] = {}

    def __init__(self, username: str, password: str) -> None:
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

    def sync(self, tables: TableLiteral | list[TableLiteral] | None = None):
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
        output: dict[TableLiteral, list[dict]] = defaultdict(list)
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
                table = s["table_name"]
                synced_at = s["last_synchronized_at"]
                sync_times[table] = synced_at
                self._sync_times[table] = synced_at
            complete = json["_complete"]
            for table, response_data in json.items():
                if table in tables:
                    output[table].extend(response_data)
        for table_name, data in output.items():
            if table_name in _ALL_TABLES:
                cols = list(data[0].keys())
                df = pd.DataFrame.from_dict(
                    {i: d for i, d in enumerate(data)}, orient="index", columns=cols
                )
                # Ensure any uuid columns are all uppercase
                for col in [c for c in df.columns if "uuid" in c]:
                    df[col] = df[col].str.upper()
                if table_name not in self.tables:
                    continue
                if table_name in _INDEX_COLS:
                    # Set the index to the index col(s), update any existing rows,
                    # then add any that arent in the index
                    table = self.tables[table_name].set_index(_INDEX_COLS[table_name])
                    df = df.set_index(_INDEX_COLS[table_name])
                    table.update(df)
                    df = pd.concat(
                        [table, df.loc[df.index.difference(table.index)]],
                        verify_integrity=True,
                    ).reset_index()
                else:
                    table = self.tables[table_name]
                    df = pd.concat([table, df], ignore_index=True)
                # Update needs to only
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
    ) -> None:
        COLOURS = {
            "red": "FF0000",
            "orange": "FF8000",
            "green": "00CC00",
            "blue": "0000FF",
            "purple": "8000FF",
            "pink": "FF00FF",
            "grey": "808080",
        }
        circuit_id = uuid.uuid4()
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
        )
        response.raise_for_status()
        payload = response.json()

    def add_climb_to_circuit(self, climb_id: str, circuit_id: str) -> None:
        # Get current circuits that this climb is part of
        self.sync("circuits")
        circuits_df = self.tables["circuits"]
        climb_in_circuit = circuits_df["climbs"].map(
            lambda circuit: any(climb["uuid"] == climb_id for climb in circuit)
        )
        existing_circuit_ids = circuits_df[climb_in_circuit]["uuid"].to_list()
        # If the climb isint already in this circuit, append circuit id to the list
        # of circuits and POST to server
        if circuit_id not in existing_circuit_ids:
            response = requests.post(
                f"{self._URL}/climb_circuits/save",
                data={
                    "climb_uuid": climb_id,
                    "circuit_uuids": existing_circuit_ids + [circuit_id],
                },
            )
            response.raise_for_status()
            payload = response.json()

    def _download_db(self):
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
                for table_name, table in self.tables.items():
                    for col in [c for c in table.columns if "uuid" in c]:
                        table[col] = table[col].str.upper()
                # Also read some DB specific tables, these arent ones that we can sync from the server
                self.tables["difficulty_grades"] = pd.read_sql_query(
                    "SELECT * FROM difficulty_grades", conn
                )
                # Update sync times
                syncs = pd.read_sql_query("SELECT * FROM shared_syncs", conn)
                self._sync_times.update({t: d for t, d in syncs.apply(tuple, axis=1)})
        self._clean_tables()

    def _clean_tables(self):
        # Ensure that any UUID columns are all upper case
        for table_name, table in self.tables.items():
            for col in [c for c in table.columns if "uuid" in c]:
                table[col] = table[col].str.upper()

