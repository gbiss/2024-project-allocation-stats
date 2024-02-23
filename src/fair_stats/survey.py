from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


class BaseSurvey:
    """Abstract survey class"""

    pass


class SingleTopicSurvey(BaseSurvey):
    """A single-topic survey"""

    @staticmethod
    def from_student(
        schedule: list[ScheduleItem], student: RenaissanceMan
    ) -> "SingleTopicSurvey":
        """Create a survey from an existing RenaissanceMan student

        Ignores divisions by topic, pooling all courses into one list.

        Args:
            schedule (list[ScheduleItem]): Schedule from which student was created
            student (RenaissanceMan): Student from which survey will be created

        Returns:
            SingleTopicSurvey: A new survey object
        """
        responses = [student.valuation.independent([item]) for item in schedule]

        return SingleTopicSurvey(schedule, responses, student.total_courses)

    def __init__(
        self, schedule: list[ScheduleItem], responses: list[int], limit: int
    ) -> None:
        """A survey that groups all responses into a single topic

        Args:
            schedule (list[ScheduleItem]): Schedule from which to draw course information
            responses (list[int]): Student survey responses
            limit (int): Total courses desired
        """
        self.course_response_map = {
            schedule[i]: responses[i] for i in range(len(schedule))
        }
        self.limit = limit
