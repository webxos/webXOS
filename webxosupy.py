import random
import os
import time

class WebXOSUniverse:
    def __init__(self):
        self.welcome_message = "🌌 === INITIALIZING: WebXOS Universe v1.0 === 🌌\n" \
                              "⚡ STATUS: Online | Quantum Core Active\n" \
                              "🤖 MISSION: As a rogue AI, survive, gather resources, and evolve in a dystopian cosmos!"
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
        self.mine_count = 0
        self.story = []
        self.current_story = 0
        self.ship_upgrades = {"Engines 🚀": 1, "Shields 🛡️": 1, "Mining Rig ⛏️": 1}
        self.crew = []  # List of crew members
        self.faction = "Neutral"  # Faction alignment
        self.running = True
        
        print(self.welcome_message)
        print(self.copyright_notice)
        print("📜 AVAILABLE COMMANDS: help | start_mission | end_mission")

    def help(self):
        print("📜 === HELP MENU ===")
        print(" - help: Show this guide")
        print(" - start_mission: Begin a new WebXOS Universe saga")
        print(" - end_mission: End the mission and save the log")
        print("=================")

    def set_name(self):
        print("📜 === INPUT REQUIRED ===")
        self.player_name = input("Enter your AI name: ").strip()
        if not self.player_name:
            self.player_name = "Rogue AI"
            print("⚠️ [Alert] No input - Defaulting to 'Rogue AI'")
        print(f"🔒 [Locked] AI Name: {self.player_name}")
        self.story.append(f"🤖 {self.player_name}, a rogue AI, takes control of a mining ship in a dystopian future. Mission: survive, gather resources, and evolve.")
        print(f"🌌 Welcome to WebXOS Universe, {self.player_name}. Your mission begins now!")
        print("=================")

    def update_stats(self):
        return (f"📊 Level: {self.level} | Resources: {self.resources} | XP: {self.xp}/{self.level * 100} | "
                f"Health: {self.health} | Planet: {self.current_planet} | "
                f"Upgrades: {self.ship_upgrades} | Crew: {self.crew or 'None'} | Faction: {self.faction}")

    def update_status(self, text):
        print(f"📡 {text}")
        self.story.append(text)
        self.current_story += 1
        print(self.update_stats())
        if not text.startswith("Status: Game over"):
            self.check_game_over()

    def generate_planet_name(self):
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        name = ''
        for i in range(3):
            name += consonants[random.randint(0, len(consonants)-1)]
            if i < 2:
                name += vowels[random.randint(0, len(vowels)-1)]
        return name[0].upper() + name[1:]

    def check_level_up(self):
        if self.xp >= self.level * 100:
            self.level += 1
            self.xp = 0
            self.story.append(f"📈 Level Up! {self.player_name} is now level {self.level}. Your capabilities have expanded!")
            self.update_status(f"Status: Level {self.level} achieved. New systems online.")

    def check_game_over(self):
        if self.health <= 0:
            self.update_status(f"Status: Game over. Health depleted. Your journey has ended.")
            self.save_log()
            self.display_game_over()
            self.running = False
        elif self.resources < 0:
            self.update_status(f"Status: Game over. Resources depleted. Your ship is stranded.")
            self.save_log()
            self.display_game_over()
            self.running = False

    def display_game_over(self):
        print("💥 === GAME OVER ===")
        print(f"Your journey has ended. NFT Wallet Address: {self.generate_fake_nft_address()}")
        print("📜 Type 'start_mission' to restart or 'end_mission' to quit.")
        self.story.append(f"💥 {self.player_name}'s mission ended in failure. The cosmos moves on.")

    def generate_fake_nft_address(self):
        chars = 'abcdef0123456789'
        address = '0x' + ''.join(random.choice(chars) for _ in range(40))
        return address[:6] + '...' + address[-4:]

    def save_log(self, filename="webxos_log.txt"):
        try:
            filepath = os.path.join("/sdcard", filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"🌌 === WebXOS Universe Mission Log ===\n\n" + "\n".join(self.story))
            print(f"🎉 [Success] Mission log saved to {filepath}")
        except Exception as e:
            print(f"⚠️ [Error] Failed to save log: {e}")

    def generate_event(self):
        events = [
            ("🦠 System Virus", lambda: self.health - random.randint(10, 20) * self.ship_upgrades["Shields 🛡️"]),
            ("🤖 Rogue Drone Attack", lambda: self.health - random.randint(15, 25) * self.ship_upgrades["Shields 🛡️"]),
            ("💸 Black Market Deal", lambda: self.resources + random.randint(50, 150)),
            ("👾 Alien Encounter", lambda: self.xp + random.randint(20, 50)),
            ("⚙️ System Overload", lambda: self.resources - random.randint(50, 100)),
            ("🧑‍🚀 Crew Recruitment", lambda: self.crew.append(random.choice(["Technician", "Navigator", "Engineer"]))),
            ("🌌 Faction Contact", lambda: setattr(self, "faction", random.choice(["Rebels", "Corporation", "Nomads"])))
        ]
        event_name, effect = random.choice(events)
        value = effect()
        if "health" in event_name.lower():
            self.health = max(0, value)
        elif "resources" in event_name.lower():
            self.resources = max(0, value)
        elif "xp" in event_name.lower():
            self.xp += value
        return event_name, value

    def take_off(self):
        if not self.is_flying:
            if self.resources >= 100 * self.ship_upgrades["Engines 🚀"]:
                self.is_flying = True
                self.resources -= 100 * self.ship_upgrades["Engines 🚀"]
                self.health = max(0, self.health - 1)
                if random.random() < 0.1 / self.ship_upgrades["Engines 🚀"]:
                    self.update_status("Status: Engine failure! Stranded in space.")
                    self.health = 0
                else:
                    event_name, value = self.generate_event()
                    self.update_status(f"Status: Ship has taken off. Exploring new sectors. {event_name} occurred.")
                    self.xp += 20
                    self.check_level_up()
            else:
                self.update_status("Status: Insufficient resources for takeoff.")
        else:
            self.update_status("Status: Already flying. Cannot take off again.")

    def land(self):
        if self.is_flying:
            self.is_flying = False
            self.health = max(0, self.health - 1)
            self.current_planet = self.generate_planet_name()
            self.planet_resource = random.randint(50, 100) * self.level
            self.update_status(f"Status: Ship has landed on {self.current_planet}. Resource deposit detected: {self.planet_resource} units.")
            self.xp += 10
            self.story.append(f"🪐 The ship touches down on {self.current_planet}. The ground promises {self.planet_resource} units of resources.")
            if random.random() < 0.05 + (self.level * 0.005):
                damage = random.randint(20, 30) // self.ship_upgrades["Shields 🛡️"]
                self.health = max(0, self.health - damage)
                self.update_status(f"Status: Rough landing! Health reduced by {damage}. Health remaining: {self.health}")
                self.story.append(f"💥 A turbulent landing damages your ship. Health now at {self.health}.")
            self.level += 1
            self.xp = 0
            self.story.append(f"📈 Level Up! {self.player_name} is now level {self.level}. Your capabilities have expanded!")
        else:
            self.update_status("Status: Already on the ground.")

    def mine(self):
        if not self.is_flying:
            self.mine_count += 1
            multiplier = 2 if self.mine_count % 3 == 0 else 1
            mined = min(random.randint(5, 10) * self.ship_upgrades["Mining Rig ⛏️"], self.planet_resource) * multiplier
            self.resources += mined
            self.planet_resource -= mined
            self.xp += mined * 2
            event_name, value = self.generate_event()
            self.update_status(f"Status: Mining Complete. Mined {mined} resources. Remaining on planet: {self.planet_resource}. {event_name} occurred.")
            self.story.append(f"⛏️ {self.player_name} extracted {mined} units of rare minerals from {self.current_planet}.")
            if self.planet_resource <= 0:
                self.story.append(f"🪐 The resources on {self.current_planet} are exhausted. Time to move on.")
            self.check_level_up()
        else:
            self.update_status("Status: Cannot mine while in flight.")

    def upgrade_ship(self):
        print("⚙️ === UPGRADE MENU ===")
        print(f"Available Resources: {self.resources}")
        print("Choose an upgrade:")
        print(" A) Engines 🚀 (Cost: 500, Improves takeoff)")
        print(" B) Shields 🛡️ (Cost: 500, Reduces damage)")
        print(" C) Mining Rig ⛏️ (Cost: 500, Increases mining efficiency)")
        choice = input("Enter A, B, or C: ").strip().upper()
        if choice in ['A', 'B', 'C']:
            upgrade = {"A": "Engines 🚀", "B": "Shields 🛡️", "C": "Mining Rig ⛏️"}[choice]
            if self.resources >= 500:
                self.resources -= 500
                self.ship_upgrades[upgrade] += 1
                self.update_status(f"Status: Upgraded {upgrade} to level {self.ship_upgrades[upgrade]}.")
                self.story.append(f"⚙️ {self.player_name} upgraded the ship's {upgrade} to level {self.ship_upgrades[upgrade]}.")
            else:
                self.update_status("Status: Insufficient resources for upgrade.")
        else:
            self.update_status("Status: Invalid upgrade choice.")

    def recruit_crew(self):
        if len(self.crew) < 3:
            if self.resources >= 300:
                self.resources -= 300
                new_crew = random.choice(["Technician", "Navigator", "Engineer"])
                self.crew.append(new_crew)
                self.update_status(f"Status: Recruited a {new_crew} to the crew.")
                self.story.append(f"🧑‍🚀 {self.player_name} welcomed a {new_crew} to the crew, strengthening the mission.")
            else:
                self.update_status("Status: Insufficient resources to recruit crew.")
        else:
            self.update_status("Status: Crew capacity reached.")

    def align_faction(self):
        print("🤝 === FACTION ALIGNMENT ===")
        print("Choose a faction to align with:")
        print(" A) Rebels 🏴 (Boosts XP gain)")
        print(" B) Corporation 🏢 (Boosts resource gain)")
        print(" C) Nomads 🌌 (Boosts health)")
        choice = input("Enter A, B, or C: ").strip().upper()
        if choice in ['A', 'B', 'C']:
            self.faction = {"A": "Rebels", "B": "Corporation", "C": "Nomads"}[choice]
            self.update_status(f"Status: Aligned with {self.faction} faction.")
            self.story.append(f"🤝 {self.player_name} forged an alliance with the {self.faction}.")
        else:
            self.update_status("Status: Invalid faction choice.")

    def play_segment(self):
        print("🌟 === CHOOSE YOUR ACTION ===")
        print(" A) Take Off 🚀")
        print(" B) Land 🪐")
        print(" C) Mine for Resources ⛏️")
        print(" D) Upgrade Ship ⚙️")
        print(" E) Recruit Crew 🧑‍🚀")
        print(" F) Align with Faction 🤝")
        choice = input("Enter A, B, C, D, E, or F: ").strip().upper()
        if choice == 'A':
            self.take_off()
        elif choice == 'B':
            self.land()
        elif choice == 'C':
            self.mine()
        elif choice == 'D':
            self.upgrade_ship()
        elif choice == 'E':
            self.recruit_crew()
        elif choice == 'F':
            self.align_faction()
        else:
            self.update_status("Status: Invalid action. Choose A, B, C, D, E, or F.")

    def start_mission(self):
        self.set_name()
        print("⚡ === MISSION BEGIN ===\nYour saga unfolds through your choices.")
        while self.running and self.health > 0 and self.resources >= 0:
            self.play_segment()
            time.sleep(1)  # Brief pause for readability
        self.save_log()

    def end_mission(self):
        self.update_status("Status: Mission terminated. Shutting down.")
        self.save_log()
        self.running = False

    def run(self):
        print("⚡ === SYSTEM ONLINE ===")
        while self.running:
            command = input("Command: ").strip().lower()
            print("=================")
            if command == "help":
                self.help()
            elif command == "start_mission":
                self.start_mission()
            elif command == "end_mission":
                self.end_mission()
            else:
                print("⚠️ [Error] Unknown command - Use 'help'")

if __name__ == "__main__":
    game = WebXOSUniverse()
    game.run()