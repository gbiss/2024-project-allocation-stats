import pytest
from fair.feature import BaseFeature, Course, Weekday, Section, Slot
from fair.item import ScheduleItem


@pytest.fixture
def course_domain():
    return ["250", "301", "611"]


@pytest.fixture
def course(course_domain: list[int]):
    return Course(course_domain)


@pytest.fixture
def slot():
    return Slot([1, 2, 3, 4, 5, 6, 7], [(1, 2), (2, 3), (4, 5), (6, 7)])


@pytest.fixture
def weekday():
    return Weekday()


@pytest.fixture
def section():
    return Section([1, 2, 3])


@pytest.fixture
def features(course: Course, slot: Slot, weekday: Weekday, section: Section):
    return [course, slot, weekday, section]


@pytest.fixture
def schedule_item250(features: list[BaseFeature]):
    return ScheduleItem(features, ["250", (1, 2), ("Mon",), 1], 0)


@pytest.fixture
def schedule_item250_2(features: list[BaseFeature]):
    return ScheduleItem(features, ["250", (4, 5), ("Mon",), 2], 1)


@pytest.fixture
def schedule_item301(features: list[BaseFeature]):
    return ScheduleItem(features, ["301", (2, 3), ("Mon",), 1], 2)


@pytest.fixture
def schedule_item301_2(features: list[BaseFeature]):
    return ScheduleItem(features, ["301", (4, 5), ("Mon",), 1], 3)


@pytest.fixture
def schedule_item611(features: list[BaseFeature]):
    return ScheduleItem(features, ["611", (4, 5), ("Mon",), 1], 4)


@pytest.fixture
def all_items(
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item301, schedule_item611]
