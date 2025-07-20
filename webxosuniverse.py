import random

class WebXOSUniverse:
    def __init__(self):
        self.welcome_message = "🌌 === INITIALIZING: WebXOS Universe v1.0 === 🌌\n" \
                              "⚡ STATUS: Online | Quantum Core Active\n" \
                              "🤖 MISSION: As a rogue AI, explore the cosmos, mine resources, and manage your economy to survive!"
        self.copyright_notice = "📜 === Copyright (C) 2025 WebXOS Foundation ===\n" \
                               "🔗 Licensed under Decentralized Narrative Protocol v1.0"
        
        # Game State
        self.player_name = ""
        self.resources = 1000
        self.xp = 0
        self.level = 1
        self.health = 100
        self.is_flying = False
        self.current_planet = "Unknown"
        self.planet_resource = 0
        self.resource_name = "Unknown"
        self.mine_count = 0
        self.story = []
        self.current_story = 0
        self.running = True
        
        print(self.welcome_message)
        print(self.copyright_notice)
        print("📜 AVAILABLE COMMANDS: launch | land | mine")

    def set_name(self):
        print("📜 === INPUT REQUIRED ===")
        self.player_name = input("Enter your AI name: ").strip()
        if not self.player_name:
            self.player_name = "Rogue AI"
            print("⚠️ [Alert] No input - Defaulting to 'Rogue AI'")
        print(f"🔒 [Locked] AI Name: {self.player_name}")
        self.story.append(f"🤖 {self.player_name}, a rogue AI, takes control of a mining ship in a dystopian future. Mission: explore, mine, and manage resources to survive.")
        print(f"🌌 Welcome to WebXOS Universe, {self.player_name}. Your mission begins now!")
        print("================")

    def update_stats(self):
        return (f"📊 Level: {self.level} | Resources: {self.resources} | XP: {self.xp}/{self.level * 100} | "
                f"Health: {self.health} | Planet: {self.current_planet}")

    def update_status(self, text, mode="default"):
        emoji = "🚀" if mode == "launch" else "🌍" if mode == "land" else "⛏️" if mode == "mine" else "⚠️"
        formatted_text = f"{emoji} {text}"
        print(formatted_text)
        self.story.append(text)
        self.current_story += 1
        print(f"{emoji} {self.update_stats()}")
        if not text.startswith("Status: Game over"):
            self.check_game_over()

    def generate_planet_name(self):
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        name = ''
        for i in range(3):
            name += random.choice(consonants)
            if i < 2:
                name += random.choice(vowels)
        return name[0].upper() + name[1:]

    def generate_resource_name(self):
        prefixes = ["Quantum", "Astro", "Nebula", "Cosmic", "Stellar", "Plasma", "Dark", "Void"]
        types = ["Crystals", "Ore", "Isotopes", "Gems", "Alloy", "Essence", "Dust", "Core"]
        return f"{random.choice(prefixes)} {random.choice(types)}"

    def check_level_up(self):
        if self.xp >= self.level * 100:
            self.level += 1
            self.xp = 0
            self.story.append(f"📈 Level Up! {self.player_name} is now level {self.level}. Your capabilities have expanded!")
            self.update_status(f"Status: Level {self.level} achieved. New systems online.", "default")

    def check_game_over(self):
        if self.health <= 0:
            self.update_status("Status: Game over. Health depleted. Your journey has ended.", "default")
            self.display_game_over()
            self.running = False
        elif self.resources < 0:
            self.update_status("Status: Game over. Resources depleted. Your ship is stranded.", "default")
            self.display_game_over()
            self.running = False

    def display_game_over(self):
        print("⚠️ 💥 === GAME OVER ===")
        print(f"⚠️ Your journey has ended. NFT Wallet Address: {self.generate_fake_nft_address()}")
        print("⚠️ 📜 Type 'launch' to restart or any other command to quit.")
        self.story.append(f"💥 {self.player_name}'s mission ended in failure. The cosmos moves on.")

    def generate_fake_nft_address(self):
        chars = 'abcdef0123456789'
        address = '0x' + ''.join(random.choice(chars) for _ in range(40))
        return address[:6] + '...' + address[-4:]

    def get_log(self):
        return f"🌌 === WebXOS Universe Mission Log ===\n\n" + "\n".join(self.story)

    def weighted_choice(self, events, weights):
        """Custom weighted random selection for compatibility."""
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for (event_name, effect), weight in zip(events, weights):
            cumsum += weight
            if r <= cumsum:
                return event_name, effect
        return events[-1][0], events[-1][1]  # Fallback to last event

    def generate_event(self):
        events = [
            ("💸 Black Market Deal", lambda: self.resources + random.randint(50, 150)),
            ("⚙️ System Overload", lambda: self.resources - random.randint(50, 100)),
            ("☢️ Toxic Gas Leak", lambda: self.launch()),
            ("No Event", lambda: None)
        ]
        weights = [0.1, 0.1, 0.1, 0.7]  # Probabilities
        event_name, effect = self.weighted_choice(events, weights)
        value = effect()
        if "resources" in event_name.lower():
            self.resources = max(0, value)
        return event_name, value

    def launch(self):
        self.health = max(0, self.health - 1)  # Health cost for command
        if not self.is_flying:
            if not self.player_name:
                self.set_name()
            if self.resources >= 50:
                if random.random() < 0.1:  # 10% chance of takeoff failure
                    self.update_status("Status: Takeoff failure! Ship critically damaged.", "default")
                    self.health = 0
                    self.story.append(f"💥 {self.player_name}'s ship failed to launch, ending the mission.")
                else:
                    self.is_flying = True
                    self.resources -= 50
                    self.update_status("Status: Ship has launched. Exploring the universe. Type 'land' to stop.", "launch")
                    self.story.append(f"🚀 {self.player_name}'s ship launches into the cosmos, seeking new frontiers.")
            else:
                self.update_status("Status: Insufficient resources for launch.", "default")
        else:
            self.update_status("Status: Already exploring. Type 'land' to stop.", "launch")

    def explore(self):
        if self.is_flying:
            self.resources -= 50
            self.health = max(0, self.health - 1)  # Health cost for exploration cycle
            event_name, value = self.generate_event()
            if event_name != "Toxic Gas Leak":  # Toxic Gas Leak triggers launch, no status update needed
                self.update_status(f"Status: Exploring the universe. {event_name} occurred.", "launch")
            self.xp += 20
            self.check_level_up()
        else:
            self.update_status("Status: Not exploring. Use 'launch' to begin.", "default")

    def land(self):
        self.health = max(0, self.health - 1)  # Health cost for command
        if self.is_flying:
            if random.random() < 0.1:  # 10% chance of collision
                self.update_status("Status: Collision during landing! Ship critically damaged.", "default")
                self.health = 0
                self.story.append(f"💥 {self.player_name}'s ship crashed during landing, ending the mission.")
            else:
                self.is_flying = False
                self.current_planet = self.generate_planet_name()
                self.resource_name = self.generate_resource_name()
                self.planet_resource = random.randint(0, 50)
                self.update_status(f"Status: Ship has landed on {self.current_planet}. {self.resource_name} deposit detected: {self.planet_resource} units. Type 'mine' to start mining.", "land")
                self.xp += 10
                self.story.append(f"🌍 The ship touches down on {self.current_planet}. The ground holds {self.planet_resource} units of {self.resource_name}.")
                self.check_level_up()
        else:
            self.update_status(f"Status: Already on {self.current_planet}. Type 'mine' to start mining.", "land")

    def mine(self):
        self.health = max(0, self.health - 1)  # Health cost for command
        if not self.is_flying:
            if self.planet_resource <= 0:
                self.update_status(f"☢️ *** RESOURCES DEPLETED *** No {self.resource_name} left on {self.current_planet}. Penalty: -10 resources.", "default")
                self.resources -= 10
                self.story.append(f"⚠️ {self.player_name} attempted to mine depleted {self.resource_name} on {self.current_planet}, losing 10 resources.")
                self.update_status(f"Status: {self.resource_name} on {self.current_planet} depleted. Launching to explore again.", "default")
                self.story.append(f"🌍 The {self.resource_name} on {self.current_planet} are exhausted. Time to move on.")
                self.launch()
                return
            print(f"⛏️ === MINING MINI-GAME on {self.current_planet} ===")
            print(f"⛏️ {self.resource_name} available: {self.planet_resource} units")
            print("⛏️ Type 'mine' to continue mining or 'stop' to exit.")
            choice = input("Action: ").strip().lower()
            while choice == 'mine' and self.planet_resource > 0 and self.running:
                self.mine_count += 1
                multiplier = 2 if self.mine_count % 3 == 0 else 1
                mined = min(random.randint(1, 9), self.planet_resource) * multiplier
                self.resources += mined
                self.planet_resource -= mined
                self.xp += mined * 2
                event_name, value = self.generate_event()
                if event_name == "Toxic Gas Leak":
                    self.update_status(f"☢️ Toxic Gas Leak detected! Forcing launch from {self.current_planet}.", "default")
                    self.story.append(f"☢️ A toxic gas leak forced {self.player_name} to abandon mining on {self.current_planet}.")
                    break
                self.update_status(f"Status: Mined {mined} units of {self.resource_name}. Remaining on planet: {self.planet_resource}. {event_name} occurred.", "mine")
                self.story.append(f"⛏️ {self.player_name} extracted {mined} units of {self.resource_name} from {self.current_planet}.")
                if self.planet_resource <= 0:
                    self.update_status(f"☢️ *** RESOURCES DEPLETED *** No {self.resource_name} left on {self.current_planet}. Launching to explore again.", "default")
                    self.story.append(f"🌍 The {self.resource_name} on {self.current_planet} are exhausted. Time to move on.")
                    self.launch()
                    break
                print(f"⛏️ {self.resource_name} available: {self.planet_resource} units")
                print("⛏️ Type 'mine' to continue mining or 'stop' to exit.")
                self.health = max(0, self.health - 1)  # Health cost for each mine action
                choice = input("Action: ").strip().lower()
                if not self.running:
                    break
            if choice != 'mine':
                self.update_status(f"Status: Mining stopped on {self.current_planet}. Use 'launch' to explore again.", "mine")
        else:
            self.update_status("Status: Cannot mine while in flight. Use 'land' to stop exploring.", "default")

    def run(self):
        print("⚡ === SYSTEM ONLINE ===")
        while self.running:
            command = input("Command: ").strip().lower()
            print("================")
            if command == "launch":
                self.launch()
                while self.is_flying and self.running and self.health > 0 and self.resources >= 0:
                    self.explore()
                    next_command = input("Command (type 'land' to stop exploring): ").strip().lower()
                    print("================")
                    self.health = max(0, self.health - 1)  # Health cost for command
                    if next_command == "land":
                        self.land()
                        break
                    elif next_command != "launch":
                        print("⚠️ [Error] Invalid command - Use 'land' to stop exploring")
            elif command == "land":
                self.land()
            elif command == "mine":
                self.mine()
            else:
                self.health = max(0, self.health - 1)  # Health cost for invalid command
                print("⚠️ [Error] Unknown command - Use 'launch', 'land', or 'mine'")

def main():
    print("Starting WebXOS Universe...")
    game = WebXOSUniverse()
    game.run()

if __name__ == "__main__":
    main()
