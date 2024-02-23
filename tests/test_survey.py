from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
from fair_stats.survey import SingleTopicSurvey


def test_single_topic_survey(schedule: list[ScheduleItem], renaissance: RenaissanceMan):
    survey = SingleTopicSurvey.from_student(schedule, renaissance)
