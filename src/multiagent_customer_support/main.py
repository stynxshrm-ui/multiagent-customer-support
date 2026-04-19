#!/usr/bin/env python
import sys
import warnings
import os

from multiagent-customer-support.crew import SaSSupportCrew


def run():
    """
    Run the crew.
    """
    inputs={
        "ticket": "I was charged twice this month. Please fix this ASAP."
    }

    # Create and run the crew
    crew = SaaSSupportCrew().support_crew()
    result = crew.kickoff(inputs=inputs)


if __name__ == "__main__":
    run()