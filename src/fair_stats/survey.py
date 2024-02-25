from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
import numpy as np


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
        self.schedule = schedule
        self.response_map = {schedule[i]: responses[i] for i in range(len(schedule))}
        self.limit = limit

    def data(self) -> np.ndarray:
        """Create data vector from responses

        Returns:
            np.ndarray: Vector of responses
        """
        return np.array([self.response_map[item] for item in self.schedule])


class Corpus:
    """A collection of surveys"""

    def __init__(self, surveys: list[BaseSurvey]):
        """
        Args:
            surveys (list[BaseSurvey]): Survey list
        """
        self.surveys = surveys

    def _valid(self):
        """Schedule items in surveys must match

        Returns:
            bool: True if schedule items match, False otherwise
        """
        if len(self.surveys) < 1:
            return False

        base = self.surveys[0]
        for survey in self.surveys[1:]:
            if len(base.schedule) != len(survey.schedule):
                return False
            for i in range(len(base.schedule)):
                if base.schedule[i] != survey.schedule[i]:
                    return False

        return True
