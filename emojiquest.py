import random
import time
import math
import re

# Class definitions
classes = {
    "Warrior": {"emoji": "🗡️", "str": 8, "int": 3, "vit": 7, "dex": 3, "hp": 120, "mp": 30},
    "Mage": {"emoji": "🧙‍♂️", "str": 3, "int": 8, "vit": 4, "dex": 3, "hp": 80, "mp": 70},
    "Ranger": {"emoji": "🏹", "str": 6, "int": 5, "vit": 5, "dex": 6, "hp": 100, "mp": 50}
}

# Enemy definitions
base_enemies = [
    {"emoji": "👾", "name": "Zargoth", "hp": 60, "attack": 10},
    {"emoji": "🕷️", "name": "Skrix", "hp": 50, "attack": 12},
    {"emoji": "🦇", "name": "Vlyth", "hp": 55, "attack": 11},
    {"emoji": "🐍", "name": "Slytheron", "hp": 80, "attack": 15},
    {"emoji": "🦂", "name": "Stingrax", "hp": 70, "attack": 18},
    {"emoji": "🤖", "name": "Mechalon", "hp": 100, "attack": 20},
    {"emoji": "💾", "name": "Datavore", "hp": 120, "attack": 25},
    {"emoji": "🖥️", "name": "Screenix", "hp": 110, "attack": 22},
    {"emoji": "🕸️", "name": "Webtron", "hp": 130, "attack": 28},
    {"emoji": "🔋", "name": "Voltrix", "hp": 140, "attack": 26},
    {"emoji": "🐉", "name": "Drakzul", "hp": 150, "attack": 30},
    {"emoji": "👻", "name": "Spectrix", "hp": 140, "attack": 32},
    {"emoji": "👽", "name": "Xenorath", "hp": 200, "attack": 40}
]

# Special quest monsters
special_monsters = [
    {"emoji": "🦁", "name": "Shadow Lion", "hp": 300, "attack": 50, "drops": {"gems": 3, "gold": 1000}},
    {"emoji": "🧝‍♀️", "name": "Corrupted Elf", "hp": 250, "attack": 45, "drops": {"herbs": 5, "gold": 800}},
    {"emoji": "🗿", "name": "Ancient Golem", "hp": 400, "attack": 60, "drops": {"ores": 4, "gold": 1500}}
]

# Item and crafting definitions
armors = [
    {"name": "None", "def": 0, "cost": 0, "materials": {}},
    {"name": "Leather Armor", "def": 25, "cost": 1000, "materials": {"herbs": 5, "ores": 2}},
    {"name": "Chain Mail", "def": 75, "cost": 5000, "materials": {"ores": 10, "gems": 1}},
    {"name": "Plate Armor", "def": 150, "cost": 15000, "materials": {"ores": 20, "gems": 3}},
    {"name": "Mythril Armor", "def": 250, "cost": 30000, "materials": {"ores": 30, "gems": 5, "herbs": 10}},
    {"name": "Dragonhide Armor", "def": 350, "cost": 50000, "materials": {"herbs": 20, "gems": 10, "ores": 15}}
]

weapons = [
    {"name": "None", "atkBonus": 0, "spellBonus": 0, "cost": 0, "materials": {}},
    {"name": "Great Staff", "atkBonus": 0, "spellBonus": 200, "cost": 25000, "materials": {"herbs": 15, "gems": 5}},
    {"name": "Great Sword", "atkBonus": 200, "spellBonus": 0, "cost": 25000, "materials": {"ores": 15, "gems": 5}},
    {"name": "Starlight Bow", "atkBonus": 150, "spellBonus": 50, "cost": 30000, "materials": {"herbs": 10, "ores": 10, "gems": 7}}
]

# Quest system
story_paths = {
    "Path of Valor": [
        {"id": 1, "name": "Slay the Shadow Lion", "description": "Defeat the Shadow Lion terrorizing the village. 🦁", "target": "Shadow Lion", "reward": {"exp": 500, "gold": 2000, "items": {"gems": 3}}, "progress": 0, "total": 1},
        {"id": 2, "name": "Gather Mystic Herbs", "description": "Collect 20 herbs for the village healer. 🌿", "target": "herbs", "reward": {"exp": 300, "gold": 1000, "items": {"ores": 5}}, "progress": 0, "total": 20},
        {"id": 3, "name": "Forge the Lightblade", "description": "Craft a Great Sword to face the darkness. ⚔️", "target": "Great Sword", "reward": {"exp": 1000, "gold": 5000}, "progress": 0, "total": 1}
    ],
    "Path of Wisdom": [
        {"id": 4, "name": "Defeat the Corrupted Elf", "description": "Stop the Corrupted Elf's dark rituals. 🧝‍♀️", "target": "Corrupted Elf", "reward": {"exp": 450, "gold": 1800, "items": {"herbs": 5}}, "progress": 0, "total": 1},
        {"id": 5, "name": "Collect Ancient Gems", "description": "Find 15 gems for the sage's ritual. 💎", "target": "gems", "reward": {"exp": 350, "gold": 1200, "items": {"herbs": 3}}, "progress": 0, "total": 15},
        {"id": 6, "name": "Craft the Starlight Bow", "description": "Craft a Starlight Bow to channel wisdom. 🏹", "target": "Starlight Bow", "reward": {"exp": 1200, "gold": 6000}, "progress": 0, "total": 1}
    ],
    "Path of Endurance": [
        {"id": 7, "name": "Destroy the Ancient Golem", "description": "Defeat the Ancient Golem guarding the ruins. 🗿", "target": "Ancient Golem", "reward": {"exp": 600, "gold": 2500, "items": {"ores": 5}}, "progress": 0, "total": 1},
        {"id": 8, "name": "Mine Rare Ores", "description": "Gather 25 ores for the blacksmith. ⛏️", "target": "ores", "reward": {"exp": 400, "gold": 1500, "items": {"gems": 2}}, "progress": 0, "total": 25},
        {"id": 9, "name": "Forge Dragonhide Armor", "description": "Craft Dragonhide Armor for ultimate defense. 🛡️", "target": "Dragonhide Armor", "reward": {"exp": 1500, "gold": 7000}, "progress": 0, "total": 1}
    ]
}

side_quests = [
    {"id": 10, "name": "Hunt 10 Zargoths", "description": "Clear 10 Zargoths from the forest. 👾", "target": "Zargoth", "reward": {"exp": 200, "gold": 500, "items": {"herbs": 2}}, "progress": 0, "total": 10},
    {"id": 11, "name": "Slay 5 Drakzuls", "description": "Defeat 5 Drakzuls to protect the town. 🐉", "target": "Drakzul", "reward": {"exp": 300, "gold": 800, "items": {"ores": 3}}, "progress": 0, "total": 5},
    {"id": 12, "name": "Collect 10 Gems", "description": "Gather 10 gems for the merchant. 💎", "target": "gems", "reward": {"exp": 250, "gold": 600, "items": {"herbs": 3}}, "progress": 0, "total": 10}
]

def print_log(message):
    """Print a formatted game message with emoji."""
    print(f"📢 {message}")

def spawn_enemy():
    """Spawn a new enemy, scaling with hero level and enemy count."""
    global enemy_count, current_enemy, hero
    enemy_count += 1
    if hero["quests"]["active"]:
        for quest in hero["quests"]["active"]:
            if quest["target"] in [m["name"] for m in special_monsters]:
                for monster in special_monsters:
                    if monster["name"] == quest["target"]:
                        return dict(monster, hp=monster["hp"], maxHp=monster["hp"])
    base_enemy = random.choice(base_enemies)
    scale = 1 + (enemy_count * 0.1)
    level_scale = 0.8 + (hero["level"] * 0.05)
    enemy = {
        "emoji": base_enemy["emoji"],
        "name": base_enemy["name"],
        "hp": int(base_enemy["hp"] * scale),
        "maxHp": int(base_enemy["hp"] * scale),
        "attack": int(base_enemy["attack"] * scale * level_scale)
    }
    if enemy_count % 10 == 0:
        enemy["hp"] *= 2
        enemy["maxHp"] *= 2
        enemy["attack"] *= 2
        enemy["emoji"] = f"👑{enemy['emoji']}"
        enemy["name"] = f"King {enemy['name']}"
        enemy["isBoss"] = True
    elif random.random() < 0.2:
        enemy["hp"] = int(enemy["hp"] * 1.5)
        enemy["maxHp"] = int(enemy["hp"] * 1.5)
        enemy["attack"] = int(enemy["attack"] * 1.5)
        enemy["emoji"] = f"⭐{enemy['emoji']}"
        enemy["name"] = f"Elite {enemy['name']}"
        enemy["isElite"] = True
    return enemy

def health_bar(hp, max_hp, length=10):
    """Generate an emoji-based health bar."""
    filled = int(hp / max_hp * length)
    empty = length - filled
    return "❤️" * filled + "🖤" * empty

def mana_bar(mp, max_mp, length=10):
    """Generate an emoji-based mana bar."""
    filled = int(mp / max_mp * length)
    empty = length - filled
    return "🔵" * filled + "🖤" * empty

def exp_bar(exp, exp_to_level, length=10):
    """Generate an emoji-based experience bar."""
    filled = int(exp / exp_to_level * length)
    empty = length - filled
    return "⭐" * filled + "🖤" * empty

def update_ui():
    """Display the game state with enhanced combat stats and stat point warning."""
    global current_enemy, hero
    print("\n=== 🕹️ Emoji Quest ===")
    print(f"👤 {hero['name']} {classes[hero['class']]['emoji']} | Level: {hero['level']} 🌟 | Gold: {hero.get('gold', 0)} 💰")
    print(f"HP: {hero['hp']}/{hero['maxHp']} {health_bar(hero['hp'], hero['maxHp'])} | MP: {hero['mp']}/{hero['maxMp']} {mana_bar(hero['mp'], hero['maxMp'])} | EXP: {int(hero['exp'])}/{hero['expToLevel']} {exp_bar(hero['exp'], hero['expToLevel'])}")
    print(f"⚔️ Combat Stats: STR: {hero['str']} 💪 | INT: {hero['int']} 🧠 | VIT: {hero['vit']} 🛡️ | DEX: {hero['dex']} 🎯 | DEF: {hero['def']} 🛡")
    print(f"🧥 Equipment: Armor: {armors[hero['armorLevel']]['name']} | Weapon: {weapons[hero['weaponLevel']]['name']}")
    print(f"🎒 Inventory: Herbs 🌿: {hero['inventory']['herbs']}, Ores ⛏️: {hero['inventory']['ores']}, Gems 💎: {hero['inventory']['gems']}")
    if hero["statPoints"] > 0:
        print("⚠️ Stat Point Available! Use 'u' to upgrade. ⚠️")
    print(f"\n=== 👹 Enemy ===")
    print(f"{current_enemy['emoji']} {current_enemy['name']} | HP: {current_enemy['hp']}/{current_enemy['maxHp']} {health_bar(current_enemy['hp'], current_enemy['maxHp'])}")
    print(f"Attack: {current_enemy['attack']} 💥")
    print(f"\n=== 📜 Active Quests ===")
    if hero["quests"]["active"]:
        for quest in hero["quests"]["active"]:
            print(f"- {quest['name']}: {quest['description']} (Progress: {quest['progress']}/{quest['total']})")
    else:
        print("No active quests.")
    if hero["quests"]["completed"]:
        print(f"\n=== 🏆 Completed Quests ===")
        for qid in hero["quests"]["completed"]:
            for path in story_paths.values():
                for q in path:
                    if q["id"] == qid:
                        print(f"- {q['name']} (Story)")
                        break
                else:
                    continue
                break
            else:
                for q in side_quests:
                    if q["id"] == qid:
                        print(f"- {q['name']} (Side)")
                        break

def parse_command(input_str):
    """Parse player input into a command ID."""
    global hero
    input_str = input_str.lower().strip()
    if re.match(r'^(a|attack|1)$', input_str):
        return 1
    elif re.match(r'^(s|spell|2)$', input_str):
        return 2
    elif re.match(r'^(i|inn|3)$', input_str):
        return 3
    elif re.match(r'^(c|craft|4)$', input_str):
        return 4
    elif re.match(r'^(q|quest|5)$', input_str):
        return 5
    elif re.match(r'^(u|upgrade|6)$', input_str) and hero["statPoints"] > 0:
        return 6
    print_log("Invalid command! Try again (e.g., 'attack', 'a', '1') 🚫")
    return 0

def player_turn(command_id):
    """Handle the player's turn based on the command ID with enhanced combat log."""
    global inn_cooldown, current_enemy, hero
    if game_state != "playing":
        print_log("Game is not active! 🚫")
        return
    if inn_cooldown:
        print_log("Resting at the inn, please wait! 🏡")
        return
    miss_chance = 0.1  # 10% chance to miss
    crit_chance = 0.1 + (hero["dex"] * 0.01)
    crit_multiplier = 1.5 if random.random() < crit_chance else 1.0
    commands = {
        1: {
            "name": "Attack",
            "emoji": "⚔️",
            "cost": 0,
            "damage": lambda: int((hero["str"] * 3 + hero["atkBonus"]) * crit_multiplier * math.sqrt(hero["level"])),
            "action": "slashes"
        },
        2: {
            "name": "Spell",
            "emoji": "✨",
            "cost": 15,
            "damage": lambda: int((75 + hero["int"] * 3 + hero["spellBonus"]) * crit_multiplier * math.sqrt(hero["level"])),
            "action": "casts"
        },
        3: {
            "name": "Inn",
            "emoji": "🏡",
            "cost": 0,
            "effect": lambda e: start_inn_timer(),
            "log": "rests at Inn 🏡"
        },
        4: {
            "name": "Craft",
            "emoji": "🔨",
            "cost": 0,
            "effect": lambda e: craft_menu(),
            "log": "opens the Forge 🔨"
        },
        5: {
            "name": "Quest",
            "emoji": "📜",
            "cost": 0,
            "effect": lambda e: quest_menu(),
            "log": "reviews Quests 📜"
        }
    }
    command = commands.get(command_id)
    if not command:
        print_log("Invalid command! Try again (e.g., 'attack', 'a', '1') 🚫")
        return
    if command["cost"] > 0 and hero["mp"] < command["cost"]:
        print_log("Not enough Mana! 🔮")
        return
    hero["mp"] -= command["cost"]
    if command_id in [1, 2]:
        if random.random() < miss_chance:
            print_log(f"{classes[hero['class']]['emoji']} {hero['name']} swings and misses! 🚫")
        else:
            damage = command["damage"]()
            crit_emoji = "💥" if crit_multiplier > 1.0 else ""
            print_log(f"{classes[hero['class']]['emoji']} {hero['name']} {command['action']} {current_enemy['name']} for {damage} damage! {crit_emoji}")
            current_enemy["hp"] -= damage
        update_ui()
        if current_enemy["hp"] <= 0:
            win_combat()
        else:
            random_enemy_turn()
    else:
        print_log(f"{classes[hero['class']]['emoji']} {command['log']}")
        command["effect"](current_enemy)
        update_ui()

def random_enemy_turn():
    """Handle the enemy's turn with enhanced combat log."""
    global enemy_count, current_enemy, inn_cooldown, game_state, hero
    if game_state != "playing" or inn_cooldown:
        return
    miss_chance = 0.1  # 10% chance to miss
    crit_chance = 0.05 + (enemy_count * 0.005)
    crit_multiplier = 1.5 if random.random() < crit_chance else 1.0
    if random.random() < 0.5:
        if random.random() < miss_chance:
            print_log(f"{current_enemy['emoji']} {current_enemy['name']} misses! 🛡️")
        else:
            damage = max(1, int((current_enemy["attack"] * crit_multiplier) - hero["def"]))
            crit_emoji = "💥" if crit_multiplier > 1.0 else ""
            print_log(f"{current_enemy['emoji']} {current_enemy['name']} strikes {hero['name']} for {damage} damage! {crit_emoji}")
            hero["hp"] -= damage
        update_ui()
        if hero["hp"] <= 0:
            game_over()
    else:
        print_log(f"{current_enemy['emoji']} {current_enemy['name']} hesitates... ⏳")
        update_ui()

def start_inn_timer():
    """Start the inn rest timer with an automatic 10-second countdown."""
    global inn_cooldown, hero
    inn_cooldown = True
    print_log("Resting at the Inn... 🏡")
    for i in range(10, -1, -1):
        print(f"\rResting: [{health_bar(i, 10, 10)}] {i}s ⏳", end="")
        time.sleep(1)
    print()  # Newline after countdown
    hero["hp"] = hero["maxHp"]
    hero["mp"] = hero["maxMp"]
    inn_cooldown = False
    print_log("Fully restored! 🌟")

def win_combat():
    """Handle victory in combat, awarding EXP, gold, and drops."""
    global current_enemy, enemy_count, hero
    if not hero or not isinstance(hero, dict) or "gold" not in hero:
        print_log("Error: Hero data corrupted. Reinitializing gold. 🚫")
        hero["gold"] = 0
    enemy_key = f"{current_enemy['emoji']} {current_enemy['name']}"
    exp_gain = int(current_enemy["maxHp"] / 2 * (1 + enemy_count * 0.05))
    gold_gain = int(current_enemy["maxHp"] * (1 + enemy_count * 0.05))
    drops = {"herbs": random.randint(0, 2), "ores": random.randint(0, 2), "gems": random.randint(0, 1)}
    if current_enemy.get("isBoss"):
        drops = {"herbs": random.randint(2, 5), "ores": random.randint(2, 5), "gems": random.randint(1, 3)}
    elif current_enemy.get("isElite"):
        drops = {"herbs": random.randint(1, 3), "ores": random.randint(1, 3), "gems": random.randint(0, 2)}
    for monster in special_monsters:
        if monster["name"] == current_enemy["name"]:
            drops = monster["drops"]
    for item, qty in drops.items():
        hero["inventory"][item] = hero["inventory"].get(item, 0) + qty
    update_quest_progress(current_enemy["name"], drops)
    print_log(f"{enemy_key} defeated! +{exp_gain} EXP ⭐, +{gold_gain} Gold 💰, Drops: {', '.join(f'{k}: {v}' for k, v in drops.items())}")
    hero["exp"] = hero.get("exp", 0) + exp_gain
    hero["gold"] = hero.get("gold", 0) + gold_gain
    if hero["exp"] >= hero["expToLevel"]:
        level_up()
    current_enemy = spawn_enemy()
    update_ui()

def update_quest_progress(target, drops):
    """Update progress for active quests with validation."""
    global hero
    for quest in hero["quests"]["active"][:]:
        if quest["target"] == target:
            quest["progress"] = min(quest["progress"] + 1, quest["total"])
        elif quest["target"] in drops:
            quest["progress"] = min(quest["progress"] + drops[quest["target"]], quest["total"])
        if quest["progress"] >= quest["total"]:
            complete_quest(quest)

def complete_quest(quest):
    """Complete a quest, awarding rewards and updating hero stats."""
    global hero
    print_log(f"Quest Completed: {quest['name']} 🎉")
    hero["exp"] = hero.get("exp", 0) + quest["reward"]["exp"]
    hero["gold"] = hero.get("gold", 0) + quest["reward"]["gold"]
    for item, qty in quest["reward"].get("items", {}).items():
        hero["inventory"][item] = hero["inventory"].get(item, 0) + qty
    hero["quests"]["active"] = [q for q in hero["quests"]["active"] if q["id"] != quest["id"]]
    hero["quests"]["completed"].append(quest["id"])
    if hero["exp"] >= hero["expToLevel"]:
        level_up()
    print_log(f"Rewards: +{quest['reward']['exp']} EXP ⭐, +{quest['reward']['gold']} Gold 💰, Items: {', '.join(f'{k}: {v}' for k, v in quest['reward'].get('items', {}).items()) or 'None'}")

def level_up():
    """Level up the hero, increasing stats and stat points."""
    global hero
    hero["level"] += 1
    hero["exp"] -= hero["expToLevel"]
    hero["expToLevel"] = int(hero["expToLevel"] * 1.5)
    hero["statPoints"] += 1
    hero["maxHp"] += 20 + hero["vit"] * 2
    hero["maxMp"] += 10
    hero["hp"] = min(hero["hp"], hero["maxHp"])
    hero["mp"] = min(hero["mp"], hero["maxMp"])
    print_log(f"Level Up! Now Level {hero['level']}. +1 Stat Point! 🌟")

def upgrade_stat(stat):
    """Upgrade a hero stat using a stat point with class-specific bonuses."""
    global hero
    if hero["statPoints"] > 0:
        class_bonuses = {
            "Warrior": {"str": 2, "int": 1, "vit": 1, "dex": 1},
            "Mage": {"str": 1, "int": 2, "vit": 1, "dex": 1},
            "Ranger": {"str": 1, "int": 1, "vit": 1, "dex": 2}
        }
        increment = class_bonuses[hero["class"]][stat]
        hero[stat] += increment
        hero["statPoints"] -= 1
        print_log(f"+{increment} {stat.upper()}! 🎯")
        update_ui()

def craft_menu():
    """Display the crafting menu with item previews and confirmation."""
    global hero
    print("\n🔨 Forge Menu:")
    print("=== Armors ===")
    for i, armor in enumerate(armors[1:], 1):
        materials = ", ".join(f"{k}: {v} (Have: {hero['inventory'].get(k, 0)})" for k, v in armor["materials"].items())
        current_def = armors[hero["armorLevel"]]["def"]
        gain = armor["def"] - current_def if hero["armorLevel"] < i else "Owned"
        print(f"{i}. {armor['name']} (DEF: {armor['def']} [+{gain}]) - {armor['cost']} Gold (Have: {hero.get('gold', 0)}), {materials}")
    print("\n=== Weapons ===")
    for i, weapon in enumerate(weapons[1:], len(armors)):
        materials = ", ".join(f"{k}: {v} (Have: {hero['inventory'].get(k, 0)})" for k, v in weapon["materials"].items())
        bonuses = f"ATK +{weapon['atkBonus']}, Spell +{weapon['spellBonus']}"
        owned = "Owned" if hero["weaponLevel"] == i - len(armors) + 1 else ""
        print(f"{i}. {weapon['name']} ({bonuses}) {owned} - {weapon['cost']} Gold (Have: {hero.get('gold', 0)}), {materials}")
    print(f"{len(armors) + len(weapons)}. Exit 🚪")
    try:
        choice = int(input(f"Select an item to craft (1-{len(armors) + len(weapons)}): ") or "0")
        if 1 <= choice < len(armors):
            confirm = input(f"Confirm crafting {armors[choice]['name']}? (y/n): ").lower().strip()
            if confirm == 'y':
                craft_armor(choice)
            else:
                print_log("Crafting cancelled. 🚫")
        elif len(armors) <= choice < len(armors) + len(weapons):
            confirm = input(f"Confirm crafting {weapons[choice - len(armors) + 1]['name']}? (y/n): ").lower().strip()
            if confirm == 'y':
                craft_weapon(choice - len(armors) + 1)
            else:
                print_log("Crafting cancelled. 🚫")
        else:
            print_log("Exiting Forge... 🚪")
    except ValueError:
        print_log("Invalid input! Exiting Forge... 🚫")

def craft_armor(level):
    """Craft a new armor with validation."""
    global hero
    armor = armors[level]
    if hero["armorLevel"] >= level:
        print_log("You already have better or equal armor! 🛡️")
        return
    if hero.get("gold", 0) < armor["cost"]:
        print_log("Not enough Gold! 💰")
        return
    for item, qty in armor["materials"].items():
        if hero["inventory"].get(item, 0) < qty:
            print_log(f"Not enough {item}! Need {qty}, Have {hero['inventory'].get(item, 0)}. 🚫")
            return
    hero["gold"] = hero.get("gold", 0) - armor["cost"]
    for item, qty in armor["materials"].items():
        hero["inventory"][item] -= qty
    hero["def"] = armor["def"]
    hero["armorLevel"] = level
    print_log(f"Crafted {armor['name']}! DEF increased to {hero['def']} 🛡️")
    update_quest_progress(armor["name"], {})

def craft_weapon(index):
    """Craft a new weapon with validation."""
    global hero
    weapon = weapons[index]
    if hero["weaponLevel"] == index:
        print_log(f"You already own the {weapon['name']}! 🚫")
        return
    if hero.get("gold", 0) < weapon["cost"]:
        print_log("Not enough Gold! 💰")
        return
    for item, qty in weapon["materials"].items():
        if hero["inventory"].get(item, 0) < qty:
            print_log(f"Not enough {item}! Need {qty}, Have {hero['inventory'].get(item, 0)}. 🚫")
            return
    hero["gold"] = hero.get("gold", 0) - weapon["cost"]
    for item, qty in weapon["materials"].items():
        hero["inventory"][item] -= qty
    hero["atkBonus"] = weapon["atkBonus"]
    hero["spellBonus"] = weapon["spellBonus"]
    hero["weaponLevel"] = index
    print_log(f"Crafted {weapon['name']}! Bonuses: ATK +{weapon['atkBonus']}, Spell +{weapon['spellBonus']} 🌟")
    update_quest_progress(weapon["name"], {})

def quest_menu():
    """Display the quest menu with enhanced options."""
    global hero
    print("\n📜 Quest Menu:")
    print("1. Choose Story Path 🌍")
    print("2. Accept Side Quest 📋")
    print("3. View Active Quests 📝")
    print("4. View Completed Quests 🏆")
    print("5. Exit 🚪")
    try:
        choice = int(input("Select an option (1-5): ") or "0")
        if choice == 1:
            choose_story_path()
        elif choice == 2:
            accept_side_quest()
        elif choice == 3:
            update_ui()
        elif choice == 4:
            print("\n=== 🏆 Completed Quests ===")
            if hero["quests"]["completed"]:
                for qid in hero["quests"]["completed"]:
                    for path in story_paths.values():
                        for q in path:
                            if q["id"] == qid:
                                print(f"- {q['name']}: {q['description']}")
                                break
                        else:
                            continue
                        break
                    else:
                        for q in side_quests:
                            if q["id"] == qid:
                                print(f"- {q['name']}: {q['description']}")
                                break
            else:
                print("No quests completed yet.")
        else:
            print_log("Exiting Quest Menu... 🚪")
    except ValueError:
        print_log("Invalid input! Exiting Quest Menu... 🚫")

def choose_story_path():
    """Allow the player to choose a story path with validation."""
    global hero
    if hero["quests"]["active"] and any(q["id"] in [1, 4, 7] for q in hero["quests"]["active"]):
        print_log("You are already on a story path! Complete it first. 🚫")
        return
    print("\nChoose a Story Path 🌍:")
    for i, path in enumerate(story_paths.keys(), 1):
        print(f"{i}. {path}: {story_paths[path][0]['description']}")
    try:
        choice = int(input("Select a path (1-3): ") or "0")
        paths = list(story_paths.keys())
        if 1 <= choice <= 3:
            path = paths[choice - 1]
            hero["quests"]["active"].append(story_paths[path][0])
            print_log(f"Started {path}: {story_paths[path][0]['name']} 🌟")
        else:
            print_log("Invalid choice! 🚫")
    except ValueError:
        print_log("Invalid input! Exiting... 🚫")

def accept_side_quest():
    """Allow the player to accept a side quest with validation."""
    global hero
    available_quests = [q for q in side_quests if q["id"] not in hero["quests"]["completed"] and q["id"] not in [q["id"] for q in hero["quests"]["active"]]]
    if not available_quests:
        print_log("No side quests available! 🚫")
        return
    print("\nAvailable Side Quests 📋:")
    for i, quest in enumerate(available_quests, 1):
        print(f"{i}. {quest['name']}: {quest['description']}")
    try:
        choice = int(input(f"Select a quest (1-{len(available_quests)}): ") or "0")
        if 1 <= choice <= len(available_quests):
            hero["quests"]["active"].append(available_quests[choice - 1])
            print_log(f"Accepted quest: {available_quests[choice - 1]['name']} 🌟")
        else:
            print_log("Invalid choice! 🚫")
    except ValueError:
        print_log("Invalid input! Exiting... 🚫")

def game_over():
    """Handle game over, displaying stats and WEBXOS code."""
    global game_state, hero
    game_state = "over"
    print_log("System Crash! Game Over. 💀")
    if not hero or not isinstance(hero, dict):
        print_log("Error: Hero data corrupted. Cannot generate summary. 🚫")
        return
    elapsed = int(time.time() - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    game_data = (
        f"Game:Emoji Quest;Date:{time.strftime('%Y-%m-%d')};Time:{time.strftime('%H:%M:%S')};"
        f"Level:{hero.get('level', 1)};Gold:{hero.get('gold', 0)};TotalXP:{int(hero.get('exp', 0))};"
        f"TimeElapsed:{minutes}m{seconds}s"
    )
    webxos_code = encode_webxos(game_data)
    print_log(
        f"Level: {hero.get('level', 1)} 🎖️ | Gold: {hero.get('gold', 0)} 💰 | "
        f"Total XP: {int(hero.get('exp', 0))} ⭐ | Time: {minutes}m {seconds}s ⏳"
    )
    print_log(f"WEBXOS Code: {webxos_code} 🔒")

def encode_webxos(message):
    """Encode game data into a WEBXOS code."""
    timestamp = hex(int(time.time()))[2:].upper().zfill(8)
    message_hex = "".join(hex(ord(c))[2:].zfill(2) for c in message).upper()
    random1 = hex(random.randint(0, 0xFFFFFF))[2:].upper().zfill(6)
    random2 = hex(random.randint(0, 0xFFFFFF))[2:].upper().zfill(6)
    checksum = hex(len(message) * 17)[2:].upper().zfill(4)
    return f"WEBXOS-{timestamp}-{message_hex}-{random1}-{random2}-{checksum}"

def parse_class_input(input_str):
    """Parse class selection input."""
    input_str = input_str.lower().strip()
    if re.match(r'^(w|warrior|1)$', input_str):
        return "Warrior"
    elif re.match(r'^(m|mage|2)$', input_str):
        return "Mage"
    elif re.match(r'^(r|ranger|archer|3)$', input_str):
        return "Ranger"
    return None

def main():
    """Main game loop for Emoji Quest."""
    global game_state, current_enemy, hero, enemy_count, inn_cooldown, start_time
    # Initialize game state
    hero = {
        "name": "",
        "class": "",
        "level": 1,
        "hp": 100,
        "maxHp": 100,
        "mp": 50,
        "maxMp": 50,
        "exp": 0,
        "expToLevel": 50,
        "str": 5,
        "int": 5,
        "vit": 5,
        "dex": 5,
        "def": 0,
        "gold": 0,
        "statPoints": 0,
        "armorLevel": 0,
        "weaponLevel": 0,
        "atkBonus": 0,
        "spellBonus": 0,
        "inventory": {"herbs": 0, "ores": 0, "gems": 0},
        "quests": {"active": [], "completed": []}
    }
    game_state = "setup"
    enemy_count = 0
    current_enemy = None
    inn_cooldown = False
    start_time = time.time()

    print("Welcome to Emoji Quest! 🎮")
    try:
        name_input = input("Enter your hero's name (letters, numbers, spaces only): ") or "Hero"
        if not re.match(r'^[a-zA-Z0-9 ]+$', name_input):
            print_log("Invalid name! Using default name 'Hero'. 🚫")
            hero["name"] = "Hero"
        else:
            hero["name"] = name_input.strip()
    except Exception as e:
        print_log(f"Error entering name: {str(e)}. Using default name 'Hero'. 🚫")
        hero["name"] = "Hero"
    print("Choose your class:")
    print("1. Warrior 🗡️ (High STR: 8, HP: 120, MP: 30, DEX: 3)")
    print("2. Mage 🧙‍♂️ (High INT: 8, HP: 80, MP: 70, DEX: 3)")
    print("3. Ranger 🏹 (High DEX: 6, STR: 6, HP: 100, MP: 50)")
    try:
        class_choice = input("Select a class (1-3, or w/m/r/archer): ") or "1"
        selected_class = parse_class_input(class_choice)
        if selected_class in classes:
            hero["class"] = selected_class
            hero["str"] = classes[hero["class"]]["str"]
            hero["int"] = classes[hero["class"]]["int"]
            hero["vit"] = classes[hero["class"]]["vit"]
            hero["dex"] = classes[hero["class"]]["dex"]
            hero["hp"] = classes[hero["class"]]["hp"]
            hero["maxHp"] = classes[hero["class"]]["hp"]
            hero["mp"] = classes[hero["class"]]["mp"]
            hero["maxMp"] = classes[hero["class"]]["mp"]
        else:
            print_log("Invalid class choice! Defaulting to Warrior. 🚫")
            hero["class"] = "Warrior"
            hero["str"] = classes["Warrior"]["str"]
            hero["int"] = classes["Warrior"]["int"]
            hero["vit"] = classes["Warrior"]["vit"]
            hero["dex"] = classes["Warrior"]["dex"]
            hero["hp"] = classes["Warrior"]["hp"]
            hero["maxHp"] = classes["Warrior"]["hp"]
            hero["mp"] = classes["Warrior"]["mp"]
            hero["maxMp"] = classes["Warrior"]["mp"]
    except Exception as e:
        print_log(f"Error selecting class: {str(e)}. Defaulting to Warrior. 🚫")
        hero["class"] = "Warrior"
        hero["str"] = classes["Warrior"]["str"]
        hero["int"] = classes["Warrior"]["int"]
        hero["vit"] = classes["Warrior"]["vit"]
        hero["dex"] = classes["Warrior"]["dex"]
        hero["hp"] = classes["Warrior"]["hp"]
        hero["maxHp"] = classes["Warrior"]["hp"]
        hero["mp"] = classes["Warrior"]["mp"]
        hero["maxMp"] = classes["Warrior"]["mp"]
    game_state = "playing"
    current_enemy = spawn_enemy()
    print_log(f"Welcome, {hero['name']} the {hero['class']} {classes[hero['class']]['emoji']}!")
    while game_state == "playing":
        if not inn_cooldown:
            update_ui()
            commands = "=== 🎮 Commands: 1. Attack (a) ⚔️ | 2. Spell (s) ✨ (15 MP) | 3. Inn (i) 🏡 | 4. Craft (c) 🔨 | 5. Quest (q) 📜"
            if hero["statPoints"] > 0:
                commands += " | 6. Upgrade (u) 🎯"
                print("⚠️ Stat Point Available! Use 'u' to upgrade. ⚠️")
            commands += " ==="
            print(commands)
            try:
                choice = input("Enter command (1-6, or a/s/i/c/q/u): ") or "0"
                command_id = parse_command(choice)
                if command_id in [1, 2, 3, 4, 5]:
                    player_turn(command_id)
                elif command_id == 6 and hero["statPoints"] > 0:
                    print("\nUpgrade Stats:")
                    print("1. STR 💪 (+Attack Damage)")
                    print("2. INT 🧠 (+Spell Damage)")
                    print("3. VIT 🛡️ (+HP, Defense)")
                    print("4. DEX 🎯 (+Crit Chance)")
                    try:
                        stat_choice = int(input("Select stat to upgrade (1-4): ") or "0")
                        if stat_choice == 1:
                            upgrade_stat("str")
                        elif stat_choice == 2:
                            upgrade_stat("int")
                        elif stat_choice == 3:
                            upgrade_stat("vit")
                        elif stat_choice == 4:
                            upgrade_stat("dex")
                        else:
                            print_log("Invalid stat choice! 🚫")
                    except ValueError:
                        print_log("Invalid stat input! 🚫")
                else:
                    print_log("Invalid command! Use numbers (1-6) or letters (a/s/i/c/q/u). 🚫")
            except ValueError:
                print_log("Please enter a valid command! 🚫")

if __name__ == "__main__":
    main()
