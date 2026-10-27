"""Per-company Oracle state, persisted between requests (plan section 4.1).

`Oracle` is constructed fresh per request with `global_month = 0`, so on its own
the pending-memory queue dies with the request and nothing written during one
cycle can mature during the next. The product path exports the Oracle's state
after every analysis and imports it before the next one for the same company,
through these two functions. The table is a cache keyed by company id; the
browser remains the founder's record of months (plan section 2.4, option a).
"""

from __future__ import annotations

import json
from typing import Any

from backend.database import connect, utc_now


def load_oracle_state(company_id: str | None) -> dict[str, Any] | None:
    if not company_id:
        return None
    with connect() as connection:
        row = connection.execute(
            "SELECT state_json FROM company_oracle_state WHERE company_id = ?",
            (company_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["state_json"])
    except (TypeError, ValueError):
        return None


def save_oracle_state(company_id: str | None, state: dict[str, Any]) -> None:
    if not company_id:
        return
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO company_oracle_state (company_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(company_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (company_id, json.dumps(state), utc_now()),
        )


def clear_oracle_state(company_id: str | None) -> None:
    if not company_id:
        return
    with connect() as connection:
        connection.execute(
            "DELETE FROM company_oracle_state WHERE company_id = ?", (company_id,)
        )
