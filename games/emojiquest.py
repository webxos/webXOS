import random
import time
import math
import re

# Game configuration
config = {
    "classes": {
        "Warrior": {"emoji": "🗡️", "str": 8, "int": 3, "vit": 7, "dex": 3, "hp": 120, "mp": 30},
        "Mage": {"emoji": "🧙‍♂️", "str": 3, "int": 8, "vit": 4, "dex": 3, "hp": 80, "mp": 70},
        "Ranger": {"emoji": "🏹", "str": 6, "int": 5, "vit": 5, "dex": 6, "hp": 100, "mp": 50}
    },
    "base_enemies": [
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
    ],
    "armors": [
        {"name": "None", "def": 0, "cost": 0, "materials": {}},
        {"name": "Leather Armor", "def": 25, "cost": 1000, "materials": {"herbs": 5, "ores": 2}},
        {"name": "Chain Mail", "def": 75, "cost": 5000, "materials": {"ores": 10, "gems": 1}},
        {"name": "Plate Armor", "def": 150, "cost": 15000, "materials": {"ores": 20, "gems": 3}},
        {"name": "Mythril Armor", "def": 250, "cost": 30000, "materials": {"ores": 30, "gems": 5, "herbs": 10}},
        {"name": "Dragonhide Armor", "def": 350, "cost": 50000, "materials": {"herbs": 20, "gems": 10, "ores": 15}}
    ],
    "weapons": [
        {"name": "None", "atkBonus": 0, "spellBonus": 0, "cost": 0, "materials": {}},
        {"name": "Great Staff", "atkBonus": 0, "spellBonus": 200, "cost": 25000, "materials": {"herbs": 15, "gems": 5}},
        {"name": "Great Sword", "atkBonus": 200, "spellBonus": 0, "cost": 25000, "materials": {"ores": 15, "gems": 5}},
        {"name": "Starlight Bow", "atkBonus": 150, "spellBonus": 50, "cost": 30000, "materials": {"herbs": 10, "ores": 10, "gems": 7}}
    ]
}

def print_log(message):
    print(f"📢 {message}")

def spawn_enemy(state):
    state["enemy_count"] += 1
    base_enemy = config["base_enemies"][state["enemy_count"] % len(config["base_enemies"])]
    scale = 1 + (state["enemy_count"] * 0.1)
    level_scale = 0.8 + (state["hero"]["level"] * 0.05)
    enemy = {
        "emoji": base_enemy["emoji"],
        "name": base_enemy["name"],
        "hp": int(base_enemy["hp"] * scale),
        "maxHp": int(base_enemy["hp"] * scale),
        "attack": int(base_enemy["attack"] * scale * level_scale)
    }
    if state["enemy_count"] % 10 == 0:
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
    filled = int(hp / max_hp * length)
    return "❤️" * filled + "🖤" * (length - filled)

def mana_bar(mp, max_mp, length=10):
    filled = int(mp / max_mp * length)
    return "🔵" * filled + "🖤" * (length - filled)

def exp_bar(exp, exp_to_level, length=10):
    filled = int(exp / exp_to_level * length)
    return "⭐" * filled + "🖤" * (length - filled)

def update_ui(state):
    hero = state["hero"]
    enemy = state["current_enemy"]
    print(f"\n=== 🕹️ Emoji Quest ===")
    print(f"👤 {hero['name']} {config['classes'][hero['class']]['emoji']} | Level: {hero['level']} 🌟 | Gold: {hero.get('gold', 0)} 💰")
    print(f"HP: {hero['hp']}/{hero['maxHp']} {health_bar(hero['hp'], hero['maxHp'])} | MP: {hero['mp']}/{hero['maxMp']} {mana_bar(hero['mp'], hero['maxMp'])} | EXP: {int(hero['exp'])}/{hero['expToLevel']} {exp_bar(hero['exp'], hero['expToLevel'])}")
    print(f"⚔️ Combat Stats: STR: {hero['str']} 💪 | INT: {hero['int']} 🧠 | VIT: {hero['vit']} 🛡️ | DEX: {hero['dex']} 🎯 | DEF: {hero['def']} 🛡")
    print(f"🧥 Equipment: Armor: {config['armors'][hero['armorLevel']]['name']} | Weapon: {config['weapons'][hero['weaponLevel']]['name']}")
    print(f"🎒 Inventory: Herbs 🌿: {hero['inventory']['herbs']}, Ores ⛏️: {hero['inventory']['ores']}, Gems 💎: {hero['inventory']['gems']}")
    if hero["statPoints"] > 0:
        print("⚠️ Stat Point Available! Use 'u' to upgrade. ⚠️")
    print(f"\n=== 👹 Enemy ===")
    print(f"{enemy['emoji']} {enemy['name']} | HP: {enemy['hp']}/{enemy['maxHp']} {health_bar(enemy['hp'], enemy['maxHp'])}")
    print(f"Attack: {enemy['attack']} 💥")

def parse_command(input_str, state):
    input_str = input_str.lower().strip()
    if input_str in ('a', 'attack', '1'):
        return 1
    if input_str in ('s', 'spell', '2'):
        return 2
    if input_str in ('i', 'inn', '3'):
        return 3
    if input_str in ('c', 'craft', '4'):
        return 4
    if input_str in ('u', 'upgrade', '6') and state["hero"]["statPoints"] > 0:
        return 6
    print_log("Invalid command! Try again (e.g., 'attack', 'a', '1') 🚫")
    return 0

def player_turn(command_id, state):
    if state["game_state"] != "playing":
        print_log("Game is not active! 🚫")
        return
    if state["inn_cooldown"]:
        print_log("Resting at the inn, please wait! 🏡")
        return
    hero = state["hero"]
    enemy = state["current_enemy"]
    miss_chance = 0.1
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
            "effect": lambda: start_inn_timer(state),
            "log": "rests at Inn 🏡"
        },
        4: {
            "name": "Craft",
            "emoji": "🔨",
            "cost": 0,
            "effect": lambda: craft_menu(state),
            "log": "opens the Forge 🔨"
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
            print_log(f"{config['classes'][hero['class']]['emoji']} {hero['name']} swings and misses! 🚫")
        else:
            damage = command["damage"]()
            crit_emoji = "💥" if crit_multiplier > 1.0 else ""
            print_log(f"{config['classes'][hero['class']]['emoji']} {hero['name']} {command['action']} {enemy['name']} for {damage} damage! {crit_emoji}")
            enemy["hp"] -= damage
        update_ui(state)
        if enemy["hp"] <= 0:
            win_combat(state)
        else:
            random_enemy_turn(state)
    else:
        print_log(f"{config['classes'][hero['class']]['emoji']} {command['log']}")
        command["effect"]()
        update_ui(state)

def random_enemy_turn(state):
    if state["game_state"] != "playing" or state["inn_cooldown"]:
        return
    hero = state["hero"]
    enemy = state["current_enemy"]
    miss_chance = 0.1
    crit_chance = 0.05 + (state["enemy_count"] * 0.005)
    crit_multiplier = 1.5 if random.random() < crit_chance else 1.0
    if random.random() < 0.5:
        if random.random() < miss_chance:
            print_log(f"{enemy['emoji']} {enemy['name']} misses! 🛡️")
        else:
            damage = max(1, int((enemy["attack"] * crit_multiplier) - hero["def"]))
            crit_emoji = "💥" if crit_multiplier > 1.0 else ""
            print_log(f"{enemy['emoji']} {enemy['name']} strikes {hero['name']} for {damage} damage! {crit_emoji}")
            hero["hp"] -= damage
        update_ui(state)
        if hero["hp"] <= 0:
            game_over(state)
    else:
        print_log(f"{enemy['emoji']} {enemy['name']} hesitates... ⏳")
        update_ui(state)

def start_inn_timer(state):
    state["inn_cooldown"] = True
    hero = state["hero"]
    print_log("Resting at the Inn... 🏡")
    for _ in range(10):
        print("❤️", end=" ", flush=True)
        time.sleep(1)
    print()
    hero["hp"] = hero["maxHp"]
    hero["mp"] = hero["maxMp"]
    state["inn_cooldown"] = False
    print_log("Fully restored! 🌟")

def win_combat(state):
    hero = state["hero"]
    enemy = state["current_enemy"]
    if not hero or not isinstance(hero, dict) or "gold" not in hero:
        print_log("Error: Hero data corrupted. Reinitializing gold. 🚫")
        hero["gold"] = 0
    enemy_key = f"{enemy['emoji']} {enemy['name']}"
    exp_gain = int(enemy["maxHp"] / 2 * (1 + state["enemy_count"] * 0.05))
    gold_gain = int(enemy["maxHp"] * (1 + state["enemy_count"] * 0.05))
    drops = {"herbs": random.randint(0, 2), "ores": random.randint(0, 2), "gems": random.randint(0, 1)}
    if enemy.get("isBoss"):
        drops = {"herbs": random.randint(2, 5), "ores": random.randint(2, 5), "gems": random.randint(1, 3)}
    elif enemy.get("isElite"):
        drops = {"herbs": random.randint(1, 3), "ores": random.randint(1, 3), "gems": random.randint(0, 2)}
    for item, qty in drops.items():
        hero["inventory"][item] = hero["inventory"].get(item, 0) + qty
    print_log(f"{enemy_key} defeated! +{exp_gain} EXP ⭐, +{gold_gain} Gold 💰, Drops: {', '.join(f'{k}: {v}' for k, v in drops.items())}")
    hero["exp"] = hero.get("exp", 0) + exp_gain
    hero["gold"] = hero.get("gold", 0) + gold_gain
    if hero["exp"] >= hero["expToLevel"]:
        level_up(state)
    state["current_enemy"] = spawn_enemy(state)
    update_ui(state)

def level_up(state):
    hero = state["hero"]
    hero["level"] += 1
    hero["exp"] -= hero["expToLevel"]
    hero["expToLevel"] = int(hero["expToLevel"] * 1.5)
    hero["statPoints"] += 1
    hero["maxHp"] += 20 + hero["vit"] * 2
    hero["maxMp"] += 10
    hero["hp"] = min(hero["hp"], hero["maxHp"])
    hero["mp"] = min(hero["mp"], hero["maxMp"])
    print_log(f"Level Up! Now Level {hero['level']}. +1 Stat Point! 🌟")

def upgrade_stat(stat, state):
    hero = state["hero"]
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
        update_ui(state)

def craft_menu(state):
    hero = state["hero"]
    print("\n🔨 Forge Menu:")
    print("=== Armors ===")
    for i, armor in enumerate(config["armors"][1:], 1):
        materials = ", ".join(f"{k}: {v} (Have: {hero['inventory'].get(k, 0)})" for k, v in armor["materials"].items())
        current_def = config["armors"][hero["armorLevel"]]["def"]
        gain = armor["def"] - current_def if hero["armorLevel"] < i else "Owned"
        print(f"{i}. {armor['name']} (DEF: {armor['def']} [+{gain}]) - {armor['cost']} Gold (Have: {hero.get('gold', 0)}), {materials}")
    print("\n=== Weapons ===")
    for i, weapon in enumerate(config["weapons"][1:], len(config["armors"])):
        materials = ", ".join(f"{k}: {v} (Have: {hero['inventory'].get(k, 0)})" for k, v in weapon["materials"].items())
        bonuses = f"ATK +{weapon['atkBonus']}, Spell +{weapon['spellBonus']}"
        owned = "Owned" if hero["weaponLevel"] == i - len(config["armors"]) + 1 else ""
        print(f"{i}. {weapon['name']} ({bonuses}) {owned} - {weapon['cost']} Gold (Have: {hero.get('gold', 0)}), {materials}")
    print(f"{len(config['armors']) + len(config['weapons'])}. Exit 🚪")
    try:
        choice = int(input(f"Select an item to craft (1-{len(config['armors']) + len(config['weapons'])}): ") or 0)
        if 1 <= choice < len(config["armors"]):
            confirm = input(f"Confirm crafting {config['armors'][choice]['name']}? (y/n): ").lower().strip()
            if confirm == 'y':
                craft_armor(choice, state)
            else:
                print_log("Crafting cancelled. 🚫")
        elif len(config["armors"]) <= choice < len(config["armors"]) + len(config["weapons"]):
            confirm = input(f"Confirm crafting {config['weapons'][choice - len(config['armors']) + 1]['name']}? (y/n): ").lower().strip()
            if confirm == 'y':
                craft_weapon(choice - len(config["armors"]) + 1, state)
            else:
                print_log("Crafting cancelled. 🚫")
        else:
            print_log("Exiting Forge... 🚪")
    except ValueError:
        print_log("Invalid input! Exiting Forge... 🚫")

def craft_armor(level, state):
    hero = state["hero"]
    armor = config["armors"][level]
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

def craft_weapon(index, state):
    hero = state["hero"]
    weapon = config["weapons"][index]
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

def game_over(state):
    state["game_state"] = "over"
    print_log("System Crash! Game Over. 💀")
    hero = state["hero"]
    if not hero or not isinstance(hero, dict):
        print_log("Error: Hero data corrupted. Cannot generate summary. 🚫")
        return
    elapsed = int(time.time() - state["start_time"])
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
    timestamp = hex(int(time.time()))[2:].upper().zfill(8)
    message_hex = "".join(hex(ord(c))[2:].zfill(2) for c in message).upper()
    random1 = hex(random.randint(0, 0xFFFFFF))[2:].upper().zfill(6)
    random2 = hex(random.randint(0, 0xFFFFFF))[2:].upper().zfill(6)
    checksum = hex(len(message) * 17)[2:].upper().zfill(4)
    return f"WEBXOS-{timestamp}-{message_hex}-{random1}-{random2}-{checksum}"

def parse_class_input(input_str):
    input_str = input_str.lower().strip()
    if input_str in ('w', 'warrior', '1'):
        return "Warrior"
    if input_str in ('m', 'mage', '2'):
        return "Mage"
    if input_str in ('r', 'ranger', 'archer', '3'):
        return "Ranger"
    return None

def main():
    state = {
        "game_state": "setup",
        "enemy_count": 0,
        "current_enemy": None,
        "inn_cooldown": False,
        "start_time": time.time(),
        "hero": {
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
            "inventory": {"herbs": 0, "ores": 0, "gems": 0}
        }
    }

    print("Welcome to Emoji Quest! 🎮")
    try:
        name_input = input("Enter your hero's name (letters, numbers, spaces only): ") or "Hero"
        if not re.match(r'^[a-zA-Z0-9 ]+$', name_input):
            print_log("Invalid name! Using default name 'Hero'. 🚫")
            state["hero"]["name"] = "Hero"
        else:
            state["hero"]["name"] = name_input.strip()
    except (EOFError, KeyboardInterrupt):
        print_log("Input interrupted! Using default name 'Hero'. 🚫")
        state["hero"]["name"] = "Hero"

    print("Choose your class:")
    print("1. Warrior 🗡️ (High STR: 8, HP: 120, MP: 30, DEX: 3)")
    print("2. Mage 🧙‍♂️ (High INT: 8, HP: 80, MP: 70, DEX: 3)")
    print("3. Ranger 🏹 (High DEX: 6, STR: 6, HP: 100, MP: 50)")
    try:
        class_choice = input("Select a class (1-3, or w/m/r/archer): ") or "1"
        selected_class = parse_class_input(class_choice)
        if selected_class in config["classes"]:
            state["hero"]["class"] = selected_class
            state["hero"]["str"] = config["classes"][selected_class]["str"]
            state["hero"]["int"] = config["classes"][selected_class]["int"]
            state["hero"]["vit"] = config["classes"][selected_class]["vit"]
            state["hero"]["dex"] = config["classes"][selected_class]["dex"]
            state["hero"]["hp"] = config["classes"][selected_class]["hp"]
            state["hero"]["maxHp"] = config["classes"][selected_class]["hp"]
            state["hero"]["mp"] = config["classes"][selected_class]["mp"]
            state["hero"]["maxMp"] = config["classes"][selected_class]["mp"]
        else:
            print_log("Invalid class choice! Defaulting to Warrior. 🚫")
            state["hero"]["class"] = "Warrior"
            state["hero"]["str"] = config["classes"]["Warrior"]["str"]
            state["hero"]["int"] = config["classes"]["Warrior"]["int"]
            state["hero"]["vit"] = config["classes"]["Warrior"]["vit"]
            state["hero"]["dex"] = config["classes"]["Warrior"]["dex"]
            state["hero"]["hp"] = config["classes"]["Warrior"]["hp"]
            state["hero"]["maxHp"] = config["classes"]["Warrior"]["hp"]
            state["hero"]["mp"] = config["classes"]["Warrior"]["mp"]
            state["hero"]["maxMp"] = config["classes"]["Warrior"]["mp"]
    except (EOFError, KeyboardInterrupt):
        print_log("Input interrupted! Defaulting to Warrior. 🚫")
        state["hero"]["class"] = "Warrior"
        state["hero"]["str"] = config["classes"]["Warrior"]["str"]
        state["hero"]["int"] = config["classes"]["Warrior"]["int"]
        state["hero"]["vit"] = config["classes"]["Warrior"]["vit"]
        state["hero"]["dex"] = config["classes"]["Warrior"]["dex"]
        state["hero"]["hp"] = config["classes"]["Warrior"]["hp"]
        state["hero"]["maxHp"] = config["classes"]["Warrior"]["hp"]
        state["hero"]["mp"] = config["classes"]["Warrior"]["mp"]
        state["hero"]["maxMp"] = config["classes"]["Warrior"]["mp"]

    state["game_state"] = "playing"
    state["current_enemy"] = spawn_enemy(state)
    print_log(f"Welcome, {state['hero']['name']} the {state['hero']['class']} {config['classes'][state['hero']['class']]['emoji']}!")
    while state["game_state"] == "playing":
        if not state["inn_cooldown"]:
            update_ui(state)
            commands = "=== 🎮 Commands: 1. Attack (a) ⚔️ | 2. Spell (s) ✨ (15 MP) | 3. Inn (i) 🏡 | 4. Craft (c) 🔨"
            if state["hero"]["statPoints"] > 0:
                commands += " | 6. Upgrade (u) 🎯"
                print("⚠️ Stat Point Available! Use 'u' to upgrade. ⚠️")
            commands += " ==="
            print(commands)
            try:
                choice = input("Enter command (1-4, 6, or a/s/i/c/u): ") or "0"
                command_id = parse_command(choice, state)
                if command_id in [1, 2, 3, 4]:
                    player_turn(command_id, state)
                elif command_id == 6 and state["hero"]["statPoints"] > 0:
                    print("\nUpgrade Stats:")
                    print("1. STR 💪 (+Attack Damage)")
                    print("2. INT 🧠 (+Spell Damage)")
                    print("3. VIT 🛡️ (+HP, Defense)")
                    print("4. DEX 🎯 (+Crit Chance)")
                    try:
                        stat_choice = int(input("Select stat to upgrade (1-4): ") or 0)
                        if stat_choice == 1:
                            upgrade_stat("str", state)
                        elif stat_choice == 2:
                            upgrade_stat("int", state)
                        elif stat_choice == 3:
                            upgrade_stat("vit", state)
                        elif stat_choice == 4:
                            upgrade_stat("dex", state)
                        else:
                            print_log("Invalid stat choice! 🚫")
                    except ValueError:
                        print_log("Invalid stat input! 🚫")
                else:
                    print_log("Invalid command! Use numbers (1-4, 6) or letters (a/s/i/c/u). 🚫")
            except (EOFError, KeyboardInterrupt):
                print_log("Input interrupted! Please enter a valid command. 🚫")

if __name__ == "__main__":
    main()
