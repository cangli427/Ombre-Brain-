from datetime import datetime, timedelta

from reminder_store import ReminderStore
from utils import LOCAL_TZ


def test_reminder_store_lists_due_items_by_round(tmp_path):
    store = ReminderStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})

    item = store.create(
        title="喝药",
        content="小雨 6/30 到 7/4 早晚各喝一次药。",
        repeat_rule="every_n_rounds",
        interval_rounds=3,
        daily_limit=0,
    )

    due = store.due(session_id="main", channel="gateway", round_id=1)
    assert [row["id"] for row in due] == [item["id"]]

    store.mark_reminded(item["id"], round_id=1, reminded_at="2026-07-03T20:00:00+08:00")

    assert store.due(session_id="main", channel="gateway", round_id=3) == []
    assert [row["id"] for row in store.due(session_id="main", channel="gateway", round_id=4)] == [item["id"]]


def test_reminder_store_snooze_moves_next_due(tmp_path):
    store = ReminderStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    item = store.create(title="复诊", content="喝完药去约针灸复诊。")

    snoozed = store.snooze(item["id"], minutes=120)

    assert snoozed["status"] == "active"
    assert snoozed["next_due_at"]
    assert store.due(session_id="main", channel="gateway", round_id=1) == []


def test_reminder_store_respects_scope_and_end_date(tmp_path):
    store = ReminderStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    active = store.create(
        title="窗口提醒",
        content="只在 bridge-a 里提醒。",
        session_id="bridge-a",
        channel="gateway",
    )
    expired = store.create(
        title="过期提醒",
        content="这条不该再提醒。",
        end_at="2026-07-02",
    )

    now = datetime(2026, 7, 3, 20, 0, tzinfo=LOCAL_TZ)

    assert [row["id"] for row in store.due(session_id="bridge-a", channel="gateway", round_id=1, now=now)] == [
        active["id"]
    ]
    assert store.due(session_id="bridge-b", channel="gateway", round_id=1, now=now) == []
    assert expired["id"] not in [row["id"] for row in store.due(session_id="", channel="gateway", round_id=1, now=now)]


def test_daily_reminder_sets_next_due_after_mark(tmp_path):
    store = ReminderStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    item = store.create(title="每日提醒", content="每天看一眼。", repeat_rule="daily")
    reminded_at = datetime.now(LOCAL_TZ).replace(microsecond=0)

    updated = store.mark_reminded(item["id"], round_id=1, reminded_at=reminded_at.isoformat())
    next_due = datetime.fromisoformat(updated["next_due_at"])

    assert next_due - reminded_at == timedelta(days=1)


def test_daily_limit_consumes_one_injection_per_day(tmp_path):
    store = ReminderStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    item = store.create(
        title="喝药",
        content="小雨今天早晚喝药。",
        repeat_rule="every_n_rounds",
        interval_rounds=1,
        daily_limit=1,
    )
    first_at = datetime(2026, 7, 3, 8, 0, tzinfo=LOCAL_TZ)
    next_day = datetime(2026, 7, 4, 8, 0, tzinfo=LOCAL_TZ)

    assert [row["id"] for row in store.due(session_id="", channel="gateway", round_id=1, now=first_at)] == [
        item["id"]
    ]

    updated = store.mark_reminded(item["id"], round_id=1, reminded_at=first_at.isoformat())

    assert updated["daily_reminder_date"] == "2026-07-03"
    assert updated["daily_reminder_count"] == 1
    assert store.due(session_id="", channel="gateway", round_id=5, now=first_at + timedelta(hours=4)) == []
    assert [row["id"] for row in store.due(session_id="", channel="gateway", round_id=6, now=next_day)] == [
        item["id"]
    ]


def test_expired_reminder_archives_into_history(tmp_path):
    store = ReminderStore({"state_dir": str(tmp_path / "state"), "buckets_dir": str(tmp_path / "buckets")})
    item = store.create(title="短期备忘", content="只到 7 月 3 日。", end_at="2026-07-03")

    archived = store.archive_expired(now=datetime(2026, 7, 4, 8, 0, tzinfo=LOCAL_TZ))

    assert archived == [item["id"]]
    assert store.get(item["id"])["status"] == "archived"
    assert store.due(session_id="", channel="gateway", round_id=1, now=datetime(2026, 7, 4, 8, 0, tzinfo=LOCAL_TZ)) == []
