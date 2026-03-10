import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Assignment 3: Supervisor Pattern - Travel Planning System

I chose a travel planner as my use case. A supervisor coordinates 3 sub-agents:
  1. Flight Agent   - searches and books flights
  2. Hotel Agent    - searches and books hotels
  3. Activity Agent - finds and books local tours

Why one agent isn't enough:
  - Each domain (airlines, hotels, activities) uses totally different APIs and data formats.
  - One agent with all these tools would have a huge prompt and be hard to maintain.
  - Each domain needs its own logic (flights = layovers/fares, hotels = room types/policies,
    activities = availability/group sizes).
  - The supervisor splits the work, keeps each agent focused, and combines results.
"""

from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from utils import model

# ---- API tools (stubs) ----

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights between two airports on a given date (ISO format)."""
    return (
        f"Found 3 flights from {origin} to {destination} on {date}:\n"
        f"  1. SV-201  06:00-08:30  SAR 850  (non-stop)\n"
        f"  2. SV-305  12:15-14:45  SAR 720  (non-stop)\n"
        f"  3. EK-412  09:00-15:20  SAR 650  (1 stop DXB)"
    )

@tool
def book_flight(flight_number: str, passenger_name: str) -> str:
    """Book a specific flight for a passenger."""
    return f"Flight {flight_number} booked for {passenger_name}. Confirmation: TRV-{hash(flight_number) % 10000:04d}"

@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """Search available hotels in a city for the given check-in/check-out dates."""
    return (
        f"Found 3 hotels in {city} ({checkin} to {checkout}):\n"
        f"  1. Hilton Garden Inn      - SAR 450/night ****\n"
        f"  2. Marriott City Center   - SAR 620/night *****\n"
        f"  3. Holiday Inn Express    - SAR 310/night ***"
    )

@tool
def book_hotel(hotel_name: str, guest_name: str, checkin: str, checkout: str) -> str:
    """Book a hotel room for a guest."""
    return f"{hotel_name} booked for {guest_name} ({checkin} to {checkout}). Confirmation: HTL-{hash(hotel_name) % 10000:04d}"

@tool
def search_activities(city: str, date: str) -> str:
    """Search tours and activities available in a city on a given date."""
    return (
        f"Found 3 activities in {city} on {date}:\n"
        f"  1. Old Town Walking Tour      - SAR 120  (3 hrs, 09:00)\n"
        f"  2. Desert Safari & BBQ Dinner  - SAR 280  (6 hrs, 15:00)\n"
        f"  3. Local Food Tasting Tour     - SAR 95   (2 hrs, 12:00)"
    )

@tool
def book_activity(activity_name: str, participant_name: str, date: str) -> str:
    """Book an activity for a participant."""
    return f"{activity_name} booked for {participant_name} on {date}. Confirmation: ACT-{hash(activity_name) % 10000:04d}"


# ---- Sub-agents ----

flight_agent = create_react_agent(
    model=model,
    tools=[search_flights, book_flight],
    prompt=(
        "You are a flight booking specialist. "
        "Parse natural language travel requests into structured flight searches. "
        "Convert relative dates (e.g. 'next Friday') into ISO format based on "
        f"today's date: {datetime.now().strftime('%Y-%m-%d')}. "
        "Always search for available flights first, then book the best option "
        "unless the user specifies a preference."
    ),
)

hotel_agent = create_react_agent(
    model=model,
    tools=[search_hotels, book_hotel],
    prompt=(
        "You are a hotel booking specialist. "
        "Parse natural language accommodation requests into structured searches. "
        "Convert relative dates into ISO format based on "
        f"today's date: {datetime.now().strftime('%Y-%m-%d')}. "
        "Always search for available hotels first, then book the best value "
        "option unless the user specifies a preference."
    ),
)

activity_agent = create_react_agent(
    model=model,
    tools=[search_activities, book_activity],
    prompt=(
        "You are a local activities and tours specialist. "
        "Help users find and book tours, excursions, and activities at their "
        "destination. Convert relative dates into ISO format based on "
        f"today's date: {datetime.now().strftime('%Y-%m-%d')}. "
        "Search for activities first, then recommend the best option."
    ),
)

# ---- Wrap sub-agents as tools so the supervisor can use them ----

@tool
def handle_flights(request: str) -> str:
    """Search and book flights using natural language.
    Use this when the user wants to find, compare, or book flights.
    """
    result = flight_agent.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content

@tool
def handle_hotels(request: str) -> str:
    """Search and book hotel accommodations using natural language.
    Use this when the user wants to find or book hotels.
    """
    result = hotel_agent.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content

@tool
def handle_activities(request: str) -> str:
    """Search and book local tours and activities using natural language.
    Use this when the user wants to explore tours or things to do.
    """
    result = activity_agent.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


# ---- Supervisor ----

supervisor = create_react_agent(
    model=model,
    tools=[handle_flights, handle_hotels, handle_activities],
    prompt=(
        "You are a personal travel planning assistant. "
        "You can search and book flights, hotels, and local activities. "
        "Break down user requests into appropriate tool calls and coordinate "
        "the results into a unified travel itinerary. "
        "When a request involves multiple actions, use multiple tools in sequence "
        "and present a clear summary at the end."
    ),
)


# ---- Demo ----

def main():
    print("=" * 70)
    print("TRAVEL PLANNING SUPERVISOR - Demo")
    print("=" * 70)

    # simple request - just a flight
    print("\n[1] Simple request: flight only")
    print("-" * 40)
    response = supervisor.invoke({
        "messages": [HumanMessage(
            content="Find me a flight from Riyadh (RUH) to Jeddah (JED) next Thursday."
        )]
    })
    print(response["messages"][-1].content)

    # complex request - full trip
    print("\n[2] Complex request: full trip planning")
    print("-" * 40)
    response = supervisor.invoke({
        "messages": [HumanMessage(
            content=(
                "Plan a weekend trip for Yazeed: "
                "fly from Riyadh to Jeddah next Thursday, "
                "book a hotel for 2 nights, "
                "and find a fun activity for Friday."
            )
        )]
    })
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
