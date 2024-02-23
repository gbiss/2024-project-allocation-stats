from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
from fair_stats.survey import SingleTopicSurvey
from fair.feature import Course


def test_single_topic_survey(
    schedule: list[ScheduleItem], student: RenaissanceMan, course: Course
):
    survey = SingleTopicSurvey.from_student(schedule, student)

    assert survey.limit == student.total_courses

    for item in schedule:
        if item.value(course) in student.preferred_courses:
            assert survey.course_response_map[item] == 1
        else:
            assert survey.course_response_map[item] == 0
