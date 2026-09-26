"""R2 org state: mirror, change detection, and the never-push-unpulled guard."""
from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from wxfuser import r2  # noqa: E402


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("R2_BUCKET", "tree60-test")
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="tree60-test")
        yield client


def test_round_trip_uploads_only_what_changed(s3, tmp_path):
    root = tmp_path / "org"
    assert r2.pull("wasatch-patrol", root, s3=s3) == 0  # cold start still counts as pulled
    (root / "state" / "pairs").mkdir(parents=True)
    (root / "state" / "pairs" / "a.parquet").write_bytes(b"one")
    (root / "site").mkdir()
    (root / "site" / "index.json").write_text("{}")
    assert r2.push("wasatch-patrol", root, s3=s3) == 2
    assert r2.push("wasatch-patrol", root, s3=s3) == 0  # nothing changed
    (root / "site" / "index.json").write_text('{"specs": []}')
    assert r2.push("wasatch-patrol", root, s3=s3) == 1

    obj = s3.get_object(Bucket="tree60-test", Key="orgs/wasatch-patrol/site/index.json")
    assert obj["ContentType"] == "application/json"
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket="tree60-test")["Contents"]}
    assert "orgs/wasatch-patrol/.r2-pulled" not in keys

    again = tmp_path / "again"
    assert r2.pull("wasatch-patrol", again, s3=s3) == 2
    assert (again / "state" / "pairs" / "a.parquet").read_bytes() == b"one"


def test_push_refuses_a_root_that_was_never_pulled(s3, tmp_path):
    root = tmp_path / "org"
    (root / "state").mkdir(parents=True)
    (root / "state" / "x.json").write_text("{}")
    with pytest.raises(r2.R2Error, match="not pulled"):
        r2.push("wasatch-patrol", root, s3=s3)


def test_a_root_pulled_for_one_org_cannot_be_pushed_as_another(s3, tmp_path):
    root = tmp_path / "org"
    r2.pull("wasatch-patrol", root, s3=s3)
    with pytest.raises(r2.R2Error, match="not pulled"):
        r2.push("other-org", root, s3=s3)


def test_org_ids_cannot_escape_their_prefix():
    with pytest.raises(r2.R2Error):
        r2.org_prefix("../public")
