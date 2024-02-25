from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
import numpy as np
from fair_stats.survey import Corpus, SingleTopicSurvey
from fair.feature import Course


def test_single_topic_survey(
    schedule: list[ScheduleItem], student: RenaissanceMan, course: Course
):
    survey = SingleTopicSurvey.from_student(schedule, student)
    preference_mask = [
        item.value(course) in student.preferred_courses for item in schedule
    ]

    assert survey.limit == student.total_courses
    assert [survey.response_map[item] for item in schedule] == preference_mask
    assert np.array_equal(survey.data(), np.array(preference_mask))


def test_corpus_validation(
    schedule: list[ScheduleItem],
    schedule2: list[ScheduleItem],
    student: RenaissanceMan,
    student2: RenaissanceMan,
    student3: RenaissanceMan,
):
    survey1 = SingleTopicSurvey.from_student(schedule, student)
    survey2 = SingleTopicSurvey.from_student(schedule, student2)
    survey3 = SingleTopicSurvey.from_student(schedule2, student3)
    corpus1 = Corpus([survey1, survey2])
    corpus2 = Corpus([survey1, survey3])

    assert corpus1._valid()
    assert not corpus2._valid()
