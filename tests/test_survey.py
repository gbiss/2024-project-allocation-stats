from fair.item import ScheduleItem
from fair_stats.survey import Survey


def test_survey(all_items: list[ScheduleItem]):
    responses = [0, 3, 6]
    limit = 2
    Survey(all_items, responses, limit)
