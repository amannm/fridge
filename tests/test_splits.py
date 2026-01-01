import unittest

from fridge.data.splits import make_split


class SplitTests(unittest.TestCase):
    def test_ac_on_night_holdout(self):
        records = [
            {"id": "a", "ac_on": 1, "fridge_on": 0, "environment": "night"},
            {"id": "b", "ac_on": 1, "fridge_on": 1, "environment": "night"},
            {"id": "c", "ac_on": 0, "fridge_on": 0, "environment": "day"},
        ]
        split = make_split(records, "ac_on_night_holdout", 0)
        self.assertCountEqual(split.val_ids, ["a", "b"])
        self.assertCountEqual(split.train_ids, ["c"])

    def test_environment_holdout(self):
        records = [
            {"id": "a", "ac_on": 1, "fridge_on": 0, "environment": "night"},
            {"id": "b", "ac_on": 1, "fridge_on": 1, "environment": "day"},
            {"id": "c", "ac_on": 0, "fridge_on": 0, "environment": "night"},
            {"id": "d", "ac_on": 0, "fridge_on": 1, "environment": "day"},
        ]
        split = make_split(records, "environment_holdout", 0)
        self.assertCountEqual(split.val_ids, ["a"])


if __name__ == "__main__":
    unittest.main()
