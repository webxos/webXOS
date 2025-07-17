import random

# Mars Trails game for Injector app
def mars_trails():
    # Game state
    state = {
        "crew": {"health": 100, "morale": 100},
        "resources": {"fuel": 100, "oxygen": 100, "food": 100},
        "distance": 0,
        "day": 1,
        "max_distance": 1000  # Distance to Mars base
    }

    def print_status(state):
        print(f"Day {state['day']} | Distance: {state['distance']}/{state['max_distance']} km")
        print(f"Crew: Health={state['crew']['health']}, Morale={state['crew']['morale']}")
        print(f"Resources: Fuel={state['resources']['fuel']}, Oxygen={state['resources']['oxygen']}, Food={state['resources']['food']}")

    def print_help():
        print("Mars Trails Commands:")
        print("  help    - Show this menu")
        print("  status  - Show crew and resource status")
        print("  trade   - Trade resources at a checkpoint")
        print("  continue - Move to the next day")
        print("  quit    - End the game")

    def handle_event(state):
        events = [
            ("Meteor shower hits!", {"fuel": -10, "health": -15}),
            ("Dust storm slows progress.", {"fuel": -5, "distance": -10}),
            ("Found abandoned supplies!", {"fuel": 10, "oxygen": 10, "food": 20}),
            ("Alien signal boosts morale!", {"morale": 15}),
            ("Rover glitch.", {"fuel": -15, "oxygen": -10})
        ]
        event = random.choice(events)
        print(f"Event: {event[0]}")
        for key, value in event[1].items():
            if key in state["resources"]:
                state["resources"][key] = max(0, state["resources"][key] + value)
            elif key in state["crew"]:
                state["crew"][key] = max(0, min(100, state["crew"][key] + value))
            elif key == "distance":
                state["distance"] = max(0, state["distance"] + value)
        return state

    def trade_resources(state):
        print("Trading Post: Trade fuel for food/oxygen (e.g., '10 food' to trade 10 fuel for 10 food)")
        trade = input("Enter trade (e.g., '10 food') or 'cancel': ").strip().lower()
        if trade == "cancel":
            return state
        try:
            amount, resource = trade.split()
            amount = int(amount)
            if resource not in ["food", "oxygen"] or amount <= 0 or state["resources"]["fuel"] < amount:
                print("Invalid trade. Need positive amount and enough fuel.")
                return state
            state["resources"]["fuel"] -= amount
            state["resources"][resource] += amount
            print(f"Traded {amount} fuel for {amount} {resource}.")
        except:
            print("Invalid input. Use format '10 food' or '10 oxygen'.")
        return state

    def check_game_over(state):
        if state["crew"]["health"] <= 0:
            print("Game Over: Crew health depleted. Mission failed.")
            return True
        if state["resources"]["fuel"] <= 0 or state["resources"]["oxygen"] <= 0 or state["resources"]["food"] <= 0:
            print("Game Over: Critical resource depleted. Mission failed.")
            return True
        if state["distance"] >= state["max_distance"]:
            print("Victory: You reached the Mars base! Mission accomplished!")
            return True
        return False

    print("Welcome to Mars Trails! Lead your crew to the Mars base.")
    print("Type 'help' for commands.")
    
    while True:
        command = input("Enter command: ").strip().lower()
        if command == "help":
            print_help()
        elif command == "status":
            print_status(state)
        elif command == "trade":
            state = trade_resources(state)
        elif command == "continue":
            state["day"] += 1
            state["distance"] += random.randint(50, 100)
            state["resources"]["fuel"] = max(0, state["resources"]["fuel"] - random.randint(5, 15))
            state["resources"]["oxygen"] = max(0, state["resources"]["oxygen"] - random.randint(5, 10))
            state["resources"]["food"] = max(0, state["resources"]["food"] - random.randint(5, 10))
            state["crew"]["health"] = max(0, state["crew"]["health"] - random.randint(0, 5))
            state["crew"]["morale"] = max(0, state["crew"]["morale"] - random.randint(0, 5))
            print_status(state)
            state = handle_event(state)
            if check_game_over(state):
                break
        elif command == "quit":
            print("Mission aborted. Game ended.")
            break
        else:
            print("Unknown command. Type 'help' for options.")

# Run the game
mars_trails()
