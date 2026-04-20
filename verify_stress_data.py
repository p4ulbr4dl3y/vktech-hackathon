import json
import os

DATA_PATH = "data/stress_test_500.json"
EXPECTED_GOLDEN_IDS = [
    "3666666666666666666",
    "3888888888888888888",
    "4111111111111111110",
    "4222222222222222221",
    "4444444444444444444",
    "4555555555555555555",
    "5111111111111111110",
    "5222222222222222221",
    "5444444444444444443",
    "5555555555555555555",
    "5888888888888888888",
    "4999999999999999999",
    "6222222222222222221",
    "3777777777777777777",
    "3999999999999999999",
    "4666666666666666666",
    "5999999999999999999",
]


def verify():
    assert os.path.exists(DATA_PATH), f"File {DATA_PATH} not found"
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    assert "messages" in data, "No 'messages' key in data"
    assert (
        len(data["messages"]) == 500
    ), f"Expected 500 messages, got {len(data['messages'])}"

    msg_ids = {m["id"] for m in data["messages"]}
    for gid in EXPECTED_GOLDEN_IDS:
        assert gid in msg_ids, f"Golden ID {gid} missing from dataset"

    print("Verification passed!")


if __name__ == "__main__":
    verify()
