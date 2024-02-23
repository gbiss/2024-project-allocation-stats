from fair.item import ScheduleItem


class Survey:

    def __init__(
        self, schedule: list[ScheduleItem], responses: list[int], limit: int
    ) -> None:
        self.schedule = schedule
        self.response = responses
        self.limit = limit
