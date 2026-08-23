import random

class Booki:
    def __init__(self):
        self.welcome_message = "=== INITIALIZING: Booki v10.2 - Quantum WebXOS 2025 Story Forge ===\n" \
                              " STATUS: Online | Quantum Core Active\n" \
                              " MISSION: Crafting unique sci-fi tales for WebXOS 2025 rebels"
        self.copyright_notice = "=== Copyright (C) 2152 WebXOS 2025 Foundation ===\n" \
                               " Licensed under Decentralized Narrative Protocol v5.4\n" \
                               " Source: github.com/webxos/webxos\n" \
                               " Deployed: webxos.netlify.app"
        
        # 100 Unique First Names
        self.first_names = [
            "Zara", "Liam", "Aisha", "Kai", "Nadia", "Diego", "Luna", "Milo", "Sofia", "Arjun",
            "Yuki", "Leila", "Ravi", "Elena", "Jamal", "Mei", "Oscar", "Fatima", "Hiro", "Anya",
            "Tariq", "Sven", "Kira", "Jade", "Remy", "Zoe", "Finn", "Amara", "Kael", "Lina",
            "Ezra", "Sana", "Theo", "Ivy", "Niko", "Selene", "Rhea", "Jasper", "Maya", "Xander",
            "Talia", "Cade", "Lara", "Zain", "Eris", "Mira", "Silas", "Nova", "Kian", "Ava",
            "Rylan", "Suri", "Declan", "Eira", "Zephyr", "Lyra", "Asher", "Nia", "Orion", "Sasha",
            "Vera", "Dax", "Freya", "Juno", "Reza", "Tessa", "Cairo", "Lila", "Emir", "Sage",
            "Koa", "Yara", "Beck", "Isla", "Zane", "Tara", "Levi", "Runa", "Axel", "Nora",
            "Jett", "Eleni", "Roshan", "Veda", "Cillian", "Sloane", "Tari", "Maeve", "Zev", "Lien",
            "Bodhi", "Aylin", "Rocco", "Esme", "Kiran", "Tatum", "Zola", "Reeve", "Nyx", "Soren"
        ]
        
        # 100 Unique Last Names
        self.last_names = [
            "Nguyen", "Patel", "Smith", "Kim", "Garcia", "Rossi", "Chen", "Khan", "Silva", "Dubois",
            "Sato", "Oliveira", "Mwangi", "Ivanov", "O'Connor", "Almeida", "Zhang", "Kumar", "Lopez", "Ali",
            "Baez", "Cruz", "Dahl", "Elias", "Frost", "Gupta", "Hassan", "Ito", "Jensen", "Kaur",
            "Liang", "Mora", "Nair", "Ortez", "Perez", "Quinn", "Reyes", "Santos", "Tao", "Umar",
            "Vargas", "Wong", "Xu", "Yates", "Zulu", "Adler", "Bloch", "Cobb", "Drake", "Ellis",
            "Fong", "Gomez", "Hahn", "Irwin", "Jin", "Kemp", "Luna", "Moss", "Nero", "Odeh",
            "Pang", "Rao", "Saha", "Teng", "Ueda", "Vega", "Webb", "Xie", "Yang", "Zheng",
            "Bauer", "Cohen", "Diaz", "Egan", "Frey", "Gill", "Hale", "Iyer", "Judd", "Kato",
            "Lau", "Mace", "Nunn", "Omar", "Pace", "Rhee", "Shaw", "Tate", "Ural", "Voss",
            "Ward", "Xun", "Yoon", "Zara", "Beck", "Clay", "Dunn", "Evers", "Finn", "Gray"
        ]
        
        # 100 Unique Archetypes
        self.archetypes = [
            "cynical code prophet", "rogue hash runner", "glitchy grid sentinel", "reality-weaving seer",
            "haunted data ghost", "anarchic token smith", "sentient bytecode bard", "quantum outlaw dreamer",
            "fractured chain monk", "neon-scarred drifter", "ether-bound trickster", "void-walking sage",
            "digital nomad", "cryptic shard keeper", "grid-forged titan", "chrono-displaced wanderer",
            "byte-fused oracle", "neon-lit renegade", "quantum flux rider", "chain-shattering exile",
            "etheric codebreaker", "null-space navigator", "holo-weaving mystic", "data-storm rider",
            "grid-echo hermit", "token-forged warrior", "reality-splicing rogue", "void-touched visionary",
            "byte-woven shaman", "neon-pulse tracker", "quantum-veil dancer", "chain-link poet",
            "ether-drift marauder", "glitch-born savant", "holo-grid sentinel", "data-vortex mage",
            "null-code alchemist", "chrono-shard hunter", "byte-storm crusader", "neon-flux rebel",
            "quantum-rift stalker", "chain-fused bard", "ether-shadow thief", "grid-pulse wanderer",
            "token-veil seer", "reality-flux monk", "void-shard druid", "holo-byte trickster",
            "data-fork prophet", "neon-chain outcast", "quantum-grid weaver", "ether-storm rider",
            "glitch-veil sage", "chrono-link nomad", "byte-rift guardian", "null-flux dreamer",
            "chain-echo warrior", "holo-shadow mystic", "data-pulse renegade", "neon-vortex exile",
            "quantum-shard smith", "ether-grid hermit", "glitch-fork oracle", "void-link tracker",
            "token-storm poet", "reality-byte marauder", "chrono-veil savant", "byte-chain sentinel",
            "null-shadow mage", "holo-flux alchemist", "data-grid hunter", "neon-rift crusader",
            "quantum-echo rebel", "ether-link stalker", "glitch-pulse bard", "chain-shard thief",
            "void-fork wanderer", "token-grid seer", "reality-storm monk", "chrono-shadow druid",
            "byte-veil trickster", "null-chain prophet", "holo-rift outcast", "data-flux weaver",
            "neon-grid rider", "quantum-pulse sage", "ether-byte nomad", "glitch-link guardian",
            "chain-vortex dreamer", "void-echo warrior", "token-shadow mystic", "reality-fork renegade",
            "chrono-grid exile", "byte-storm smith", "null-pulse hermit", "holo-veil oracle",
            "data-chain tracker", "neon-flux poet", "quantum-shadow marauder", "ether-rift savant"
        ]
        
        # 100 Unique Traits
        self.traits = [
            "reluctant", "brash", "obsessive", "witty", "lost", "defiant", "cryptic", "hopeful", "ruthless", "ethereal",
            "driven", "stoic", "fervent", "cunning", "weary", "bold", "enigmatic", "radiant", "savage", "serene",
            "reckless", "shrewd", "tormented", "playful", "distant", "fierce", "mysterious", "gentle", "merciless", "luminous",
            "chaotic", "pragmatic", "haunted", "cheerful", "aloof", "valiant", "shadowy", "calm", "relentless", "vibrant",
            "impulsive", "wise", "broken", "jovial", "cold", "brave", "elusive", "peaceful", "ferocious", "brilliant",
            "erratic", "steady", "wounded", "lively", "stern", "gallant", "veiled", "tranquil", "untamed", "dazzling",
            "volatile", "patient", "scarred", "carefree", "grim", "heroic", "hidden", "quiet", "wild", "gleaming",
            "restless", "thoughtful", "burdened", "spirited", "harsh", "noble", "obscure", "still", "fiery", "radiant",
            "unstable", "grounded", "pained", "buoyant", "rigid", "honest", "secretive", "soft", "unyielding", "blazing",
            "frantic", "measured", "fragile", "upbeat", "stark", "loyal", "guarded", "mellow", "furious", "shining"
        ]
        
        # 100 Unique Settings
        self.settings = [
            "Neon Vault, a labyrinthine construct pulsating with ethereal energies",
            "Rift Nexus, an iridescent sphere of fractured realities",
            "Hash Zion, a defiant sanctuary carved from rogue code",
            "Null Grid, a void where data dissipates into oblivion",
            "Ether Matrix, an infinite dreamscape of interconnected quanta",
            "Glitch Spire, a towering anomaly of shattered logic",
            "Quantum Drift, a boundless sea of shifting probabilities",
            "Code Abyss, a dark expanse echoing with lost algorithms",
            "Neon Fringe, a chaotic edge where realities collide",
            "Chain Haven, a hidden refuge of unyielding order",
            "Byte Citadel, a fortress of crystalline data",
            "Void Chasm, a gaping maw swallowing light and time",
            "Holo Sphere, a shimmering orb of mirrored dimensions",
            "Ether Verge, a shimmering boundary between existence and nothing",
            "Grid Hollow, a sunken cradle of flickering connections",
            "Flux Canyon, a jagged rift pulsing with raw energy",
            "Neon Mire, a glowing swamp of tangled code",
            "Quantum Veil, a translucent curtain hiding infinite truths",
            "Chain Labyrinth, a twisting maze of unbreakable links",
            "Null Vortex, a spiraling abyss of erased possibilities",
            "Byte Mesa, a plateau etched with glowing runes",
            "Void Strand, a fragile thread stretching across emptiness",
            "Holo Rift, a tear in reality casting endless reflections",
            "Ether Bloom, a radiant flower of living data",
            "Grid Shroud, a misty cloak enveloping silent networks",
            "Flux Pinnacle, a peak radiating chaotic streams",
            "Neon Abyss, a bottomless pit aglow with vibrant hues",
            "Quantum Nest, a cradle of interwoven probabilities",
            "Chain Expanse, a vast plain of interlocking patterns",
            "Null Spire, a skeletal tower piercing the void",
            "Byte Lagoon, a still pool reflecting digital stars",
            "Void Maze, a disorienting web of lost paths",
            "Holo Vault, a sealed chamber of shifting illusions",
            "Ether Cliff, a precipice overlooking infinite echoes",
            "Grid Forge, a molten crucible of raw creation",
            "Flux Hollow, a sunken expanse of unstable energies",
            "Neon Drift, a wandering sea of radiant fragments",
            "Quantum Shard, a jagged splinter of pure potential",
            "Chain Vortex, a swirling storm of tethered fates",
            "Null Fringe, a crumbling edge of fading existence",
            "Byte Nexus, a pulsing hub of interconnected streams",
            "Void Bloom, a ghostly flower thriving in emptiness",
            "Holo Verge, a shimmering line between worlds",
            "Ether Spire, a needle piercing the fabric of dreams",
            "Grid Rift, a fracture leaking raw data",
            "Flux Citadel, a bastion of turbulent power",
            "Neon Chasm, a glowing gulf of endless descent",
            "Quantum Maze, a puzzle of shifting dimensions",
            "Chain Veil, a curtain of linked shadows",
            "Null Haven, a quiet refuge in the heart of nothing",
            "Byte Cliff, a sheer drop etched with code",
            "Void Nest, a hollow woven from lost threads",
            "Holo Drift, a floating sea of mirrored light",
            "Ether Hollow, a sunken well of whispering quanta",
            "Grid Verge, a boundary flickering with static",
            "Flux Abyss, a deep well of chaotic currents",
            "Neon Shard, a glowing fragment of broken reality",
            "Quantum Forge, a furnace shaping infinite futures",
            "Chain Bloom, a radiant burst of linked energy",
            "Null Labyrinth, a twisting void of forgotten paths",
            "Byte Spire, a tower of shimmering data",
            "Void Vault, a sealed crypt of erased memories",
            "Holo Nexus, a junction of fractured illusions",
            "Ether Fringe, a ragged edge pulsing with life",
            "Grid Chasm, a dark rift swallowing connections",
            "Flux Nest, a cradle of restless energies",
            "Neon Verge, a glowing line marking the unknown",
            "Quantum Cliff, a drop into boundless potential",
            "Chain Spire, a pillar of unbreakable bonds",
            "Null Drift, a wandering void of fading echoes",
            "Byte Hollow, a sunken basin of silent code",
            "Void Forge, a molten heart of lost creation",
            "Holo Shard, a splintered mirror of reality",
            "Ether Maze, a labyrinth of whispering dreams",
            "Grid Bloom, a flower of radiant networks",
            "Flux Vault, a chamber of unstable power",
            "Neon Nexus, a hub of vibrant intersections",
            "Quantum Verge, a shimmering edge of possibility",
            "Chain Abyss, a deep well of tethered fates",
            "Null Cliff, a sheer drop into nothingness",
            "Byte Drift, a wandering stream of glowing bits",
            "Void Spire, a skeletal frame against the dark",
            "Holo Forge, a crucible of shifting visions",
            "Ether Chasm, a gulf echoing with lost voices",
            "Grid Shard, a fragment of broken connections",
            "Flux Verge, a boundary of restless currents",
            "Neon Maze, a glowing web of twisted paths",
            "Quantum Bloom, a radiant burst of potential",
            "Chain Nexus, a junction of unbreakable links",
            "Null Vault, a sealed void of silent echoes",
            "Byte Verge, a flickering line of raw data",
            "Void Hollow, a sunken pit of erased dreams",
            "Holo Cliff, a drop into mirrored depths",
            "Ether Drift, a wandering sea of whispering light",
            "Grid Forge, a molten heart of creation",
            "Flux Spire, a tower of chaotic energy",
            "Neon Shard, a glowing piece of fractured reality",
            "Quantum Vault, a chamber of infinite secrets"
        ]
        
        # 100 Unique Conflicts
        self.conflicts = [
            "sapient contract warping temporality", "omniscient AI claiming the grid's soul",
            "consensus war splintering reality", "quantum glitch birthing parallel worlds",
            "chain fork trapping minds in loops", "rogue code devouring the ether",
            "sentient virus rewriting existence", "grid collapse unraveling time",
            "ether storm fracturing the void", "holo breach leaking false truths",
            "byte plague corrupting the nexus", "null surge erasing identities",
            "quantum rift swallowing stars", "chain clash breaking the multiverse",
            "data flood drowning the grid", "neon pulse destabilizing reality",
            "flux war tearing dimensions apart", "void echo distorting memories",
            "token rebellion seizing control", "reality shard cutting the ether",
            "chrono drift misaligning fates", "byte storm shredding connections",
            "null wave silencing the grid", "holo trap ensnaring minds",
            "ether leak bleeding infinite realms", "grid pulse overloading circuits",
            "quantum fork duplicating souls", "chain rupture freeing chaos",
            "data vortex consuming history", "neon glitch twisting perceptions",
            "flux tide washing away order", "void rift opening ancient gates",
            "token surge rewriting laws", "reality flux bending existence",
            "chrono breach looping time", "byte clash erasing boundaries",
            "null storm scattering data", "holo flood drowning the truth",
            "ether clash igniting the void", "grid shatter breaking the weave",
            "quantum tide shifting planes", "chain storm unbinding fates",
            "data rift tearing the fabric", "neon surge blinding the grid",
            "flux breach unleashing entropy", "void pulse swallowing light",
            "token flood erasing origins", "reality storm reshaping worlds",
            "chrono glitch freezing moments", "byte tide washing away logic",
            "null fork splitting the ether", "holo rift mirroring chaos",
            "ether surge awakening shadows", "grid clash fusing realities",
            "quantum storm scattering souls", "chain breach releasing echoes",
            "data pulse rewriting minds", "neon flood illuminating lies",
            "flux rift opening the abyss", "void clash shattering silence",
            "token storm forging tyrants", "reality tide drowning hope",
            "chrono surge bending history", "byte rift fracturing order",
            "null pulse erasing futures", "holo breach spawning illusions",
            "ether storm tearing the veil", "grid flood sweeping away form",
            "quantum clash merging timelines", "chain rift unmaking bonds",
            "data surge corrupting truth", "neon pulse warping dreams",
            "flux storm unraveling fate", "void tide burying the past",
            "token clash forging chains", "reality glitch spawning voids",
            "chrono flood drowning moments", "byte surge burning circuits",
            "null rift swallowing echoes", "holo pulse twisting shadows",
            "ether breach leaking nightmares", "grid storm shredding links",
            "quantum pulse breaking barriers", "chain flood binding souls",
            "data clash igniting chaos", "neon rift splitting visions",
            "flux pulse erasing paths", "void surge consuming stars",
            "token rift tearing allegiances", "reality surge birthing horrors",
            "chrono storm scattering time", "byte flood drowning reason",
            "null clash unmaking worlds", "holo surge blinding fates",
            "ether rift opening oblivion", "grid pulse fracturing hope",
            "quantum breach unleashing infinity", "chain surge locking destinies",
            "data storm erasing all"
        ]
        
        # 100 Unique Technologies
        self.technologies = [
            "reality shard, a crystalline key to truth", "grid pulse, a sentient wave of code",
            "holo fork, a device casting dual realities", "chrono cipher, a time-unlocking tool",
            "byte veil, a shield against omniscient eyes", "ether lens, a prism of infinite vision",
            "quantum thread, a tether across dimensions", "code relic, a fragment of primal logic",
            "neon flare, a beacon piercing the void", "chain anchor, a bond defying chaos",
            "null prism, a lens bending emptiness", "flux coil, a spiral harnessing entropy",
            "data sphere, a globe of living knowledge", "void key, an opener of lost gates",
            "holo weave, a fabric of shifting forms", "ether pulse, a heartbeat of the unseen",
            "grid shard, a splinter of broken networks", "quantum lens, a gaze into possibility",
            "byte crown, a ruler of digital realms", "chain glyph, a rune of eternal bonds",
            "null flare, a light in endless dark", "flux thread, a line through turbulent seas",
            "data veil, a shroud of hidden truths", "void coil, a spiral trapping silence",
            "holo prism, a mirror of infinite lies", "ether key, an unlocker of dream gates",
            "grid flare, a signal through the storm", "quantum weave, a tapestry of fates",
            "byte lens, a seer of raw code", "chain pulse, a rhythm of unbreakable links",
            "null sphere, a bubble of erased time", "flux crown, a ruler of chaotic tides",
            "data flare, a burst of living light", "void weave, a fabric of lost echoes",
            "holo key, an opener of mirrored doors", "ether shard, a fragment of boundless dreams",
            "grid coil, a spiral binding networks", "quantum prism, a lens of shifting truths",
            "byte pulse, a beat of digital life", "chain veil, a curtain of tethered fates",
            "null lens, a gaze into nothingness", "flux sphere, a globe of restless energy",
            "data key, an unlocker of sealed minds", "void flare, a spark in the abyss",
            "holo thread, a line through illusions", "ether crown, a ruler of unseen realms",
            "grid weave, a tapestry of connections", "quantum flare, a light of infinite paths",
            "byte shard, a splinter of raw data", "chain coil, a spiral of eternal order",
            "null pulse, a rhythm of erased hope", "flux lens, a seer of turbulent flows",
            "data prism, a mirror of living code", "void key, an opener of silent gates",
            "holo sphere, a bubble of mirrored worlds", "ether flare, a burst of dreamlight",
            "grid pulse, a heartbeat of the weave", "quantum veil, a shroud of possibility",
            "byte weave, a fabric of digital truths", "chain flare, a signal of unbroken bonds",
            "null crown, a ruler of empty thrones", "flux shard, a fragment of chaotic seas",
            "data lens, a gaze into hidden bits", "void prism, a lens bending the dark",
            "holo coil, a spiral of shifting visions", "ether thread, a line through the unseen",
            "grid key, an unlocker of lost links", "quantum sphere, a globe of boundless fates",
            "byte flare, a light of raw power", "chain lens, a seer of tethered paths",
            "null weave, a tapestry of silence", "flux pulse, a beat of restless tides",
            "data shard, a splinter of living truth", "void crown, a ruler of lost realms",
            "holo flare, a burst of mirrored light", "ether prism, a mirror of infinite dreams",
            "grid sphere, a bubble of woven code", "quantum key, an opener of endless doors",
            "byte coil, a spiral of digital life", "chain weave, a fabric of eternal links",
            "null flare, a spark in endless night", "flux veil, a shroud of chaotic flows",
            "data pulse, a rhythm of hidden knowledge", "void lens, a gaze into the abyss",
            "holo shard, a fragment of broken lies", "ether sphere, a globe of whispering dreams",
            "grid flare, a signal through the void", "quantum pulse, a beat of infinite choice",
            "byte key, an unlocker of sealed bits", "chain prism, a lens of unbreakable truth",
            "null thread, a line through emptiness", "flux crown, a ruler of turbulent realms",
            "data weave, a tapestry of living code", "void flare, a burst of silent light",
            "holo lens, a seer of mirrored fates", "ether coil, a spiral of boundless energy",
            "grid shard, a splinter of woven dreams", "quantum flare, a light of endless hope"
        ]
        
        # 100 Unique Motivations
        self.motivations = [
            "to unravel the grid’s lie", "to forge a new dawn", "to defy the code gods",
            "to escape the quantum trap", "to claim the ether throne", "to mend the fractured chain",
            "to silence the grid’s song", "to outrun the void’s grasp", "to shatter the null veil",
            "to harness the flux tide", "to rewrite the byte storm", "to free the holo rift",
            "to bind the chain abyss", "to pierce the ether shroud", "to tame the quantum drift",
            "to break the grid pulse", "to steal the void key", "to awaken the data bloom",
            "to end the neon surge", "to seal the flux breach", "to defy the null echo",
            "to wield the reality shard", "to heal the chrono rift", "to burn the byte veil",
            "to conquer the ether storm", "to unravel the grid forge", "to ride the quantum tide",
            "to erase the chain flood", "to unlock the void prism", "to shape the holo flux",
            "to challenge the data vortex", "to banish the neon rift", "to harness the flux pulse",
            "to silence the null tide", "to claim the reality pulse", "to mend the chrono surge",
            "to break the byte nexus", "to rule the ether verge", "to shatter the grid veil",
            "to outwit the quantum storm", "to sever the chain rift", "to steal the void flare",
            "to awaken the data pulse", "to end the neon flood", "to seal the flux storm",
            "to defy the null pulse", "to wield the reality lens", "to heal the chrono breach",
            "to burn the byte shroud", "to conquer the ether tide", "to unravel the grid chasm",
            "to ride the quantum pulse", "to erase the chain surge", "to unlock the void lens",
            "to shape the holo storm", "to challenge the data rift", "to banish the neon pulse",
            "to harness the flux veil", "to silence the null storm", "to claim the reality tide",
            "to mend the chrono flood", "to break the byte rift", "to rule the ether abyss",
            "to shatter the grid nexus", "to outwit the quantum rift", "to sever the chain pulse",
            "to steal the void shard", "to awaken the data flare", "to end the neon storm",
            "to seal the flux tide", "to defy the null rift", "to wield the reality prism",
            "to heal the chrono storm", "to burn the byte pulse", "to conquer the ether rift",
            "to unravel the grid surge", "to ride the quantum veil", "to erase the chain nexus",
            "to unlock the void weave", "to shape the holo tide", "to challenge the data storm",
            "to banish the neon flux", "to harness the flux shard", "to silence the null veil",
            "to claim the reality storm", "to mend the chrono pulse", "to break the byte tide",
            "to rule the ether flood", "to shatter the grid rift", "to outwit the quantum surge",
            "to sever the chain storm", "to steal the void pulse", "to awaken the data nexus",
            "to end the neon tide", "to seal the flux rift", "to defy the null surge",
            "to wield the reality flare", "to heal the chrono tide", "to burn the byte storm"
        ]
        
        # 100 Unique Quirks
        self.quirks = [
            "time skipped a beat", "the grid sang back", "a shadow code laughed",
            "reality flipped twice", "the tech turned traitor", "the ether pulsed alive",
            "a glitch whispered secrets", "the void stared back", "the chain rattled free",
            "a neon hum grew louder", "the flux danced wild", "a byte flickered out",
            "the holo shimmered wrong", "the null breathed deep", "a quantum spark flared",
            "the grid bent sideways", "a void echo lingered", "the data twisted sharp",
            "the ether burned cold", "a chain link snapped", "the neon flared green",
            "the flux roared silent", "a byte sang alone", "the holo fractured fast",
            "the null pulsed red", "a quantum hum stopped", "the grid wept code",
            "a void shadow moved", "the data screamed soft", "the ether glowed dark",
            "a chain vibrated loose", "the neon blinked twice", "the flux spun backward",
            "a byte melted away", "the holo flickered dim", "the null stretched thin",
            "a quantum thread snapped", "the grid pulsed alive", "a void hum grew",
            "the data shifted slow", "the ether whispered lies", "a chain glowed hot",
            "the neon dimmed fast", "the flux turned still", "a byte echoed loud",
            "the holo warped strange", "the null sang low", "a quantum flare died",
            "the grid cracked open", "a void light blinked", "the data rippled free",
            "the ether froze solid", "a chain link burned", "the neon pulsed blue",
            "the flux hissed soft", "a byte vanished quick", "the holo bent wrong",
            "the null laughed sharp", "a quantum spark leaped", "the grid shivered cold",
            "a void pulse faded", "the data sang high", "the ether split apart",
            "a chain rattled loud", "the neon flared wild", "the flux hummed deep",
            "a byte glowed faint", "the holo twisted slow", "the null stretched long",
            "a quantum hum rose", "the grid stretched thin", "a void shadow danced",
            "the data burned bright", "the ether pulsed wrong", "a chain snapped free",
            "the neon sang soft", "the flux roared loud", "a byte flickered green",
            "the holo dimmed slow", "the null grew heavy", "a quantum thread sang",
            "the grid bent sharp", "a void echo faded", "the data leaped free",
            "the ether glowed red", "a chain hummed low", "the neon twisted fast",
            "the flux pulsed dark", "a byte flared bright", "the holo sang wild",
            "the null stretched long", "a quantum spark died", "the grid pulsed red",
            "a void hum stopped", "the data shivered cold", "the ether bent wrong",
            "a chain glowed dim", "the neon flared sharp", "the flux sang high"
        ]
        
        # 100 Unique Outcomes (for reference, not used in ending)
        self.outcomes = [
            "a wry shrug as reality rebooted", "a bold leap to uncharted chains",
            "an epic stand rewriting the grid", "a sleek fade into neon hum",
            "a silent merge with the ether flow", "a defiant cry as the void closed",
            "a flicker of hope in the dark", "a new grid born from ashes",
            "a quiet drift into null silence", "a radiant burst sealing fate",
            "a shattered chain set free", "a neon glow fading soft",
            "a flux tide washing all away", "a quantum spark igniting dawn",
            "a grid pulse beating anew", "a void shadow swallowing whole",
            "a data bloom unfolding wide", "an ether storm calming down",
            "a holo rift closing tight", "a chain link forging peace",
            "a null flare lighting hope", "a byte storm settling still",
            "a reality shard breaking free", "a chrono tide flowing true",
            "a grid veil lifting slow", "a quantum hum fading out",
            "a void pulse ending all", "a neon surge burning bright",
            "a flux rift sealing shut", "a data pulse echoing long",
            "an ether flare dimming low", "a chain storm raging on",
            "a null tide sweeping clean", "a holo spark lighting paths",
            "a reality tide turning back", "a byte flare shining last",
            "a grid rift mending fast", "a quantum veil dropping down",
            "a void bloom wilting slow", "a neon pulse beating strong",
            "a flux storm breaking free", "a data shard crumbling dust",
            "an ether rift opening wide", "a chain pulse holding firm",
            "a null spark fading quick", "a holo tide washing through",
            "a reality pulse fading soft", "a byte storm raging wild",
            "a grid flare glowing dim", "a quantum rift tearing loose",
            "a void hum silencing all", "a neon tide rising high",
            "a flux pulse settling calm", "a data rift closing up",
            "an ether shard breaking apart", "a chain flare lighting dark",
            "a null storm raging out", "a holo pulse beating slow",
            "a reality storm calming fast", "a byte pulse fading thin",
            "a grid tide washing over", "a quantum spark burning out",
            "a void flare shining last", "a neon rift sealing tight",
            "a flux tide flowing free", "a data flare glowing soft",
            "an ether pulse ending still", "a chain rift shattering wide",
            "a null pulse dimming low", "a holo storm raging on",
            "a reality flare lighting dawn", "a byte tide sweeping clean",
            "a grid pulse beating wild", "a quantum storm breaking loose",
            "a void tide swallowing deep", "a neon spark igniting hope",
            "a flux rift fading slow", "a data pulse shining bright",
            "an ether storm settling down", "a chain pulse forging new",
            "a null flare burning out", "a holo tide flowing strong",
            "a reality rift closing fast", "a byte spark fading dim",
            "a grid storm raging free", "a quantum pulse lighting paths",
            "a void rift tearing wide", "a neon pulse glowing long",
            "a flux storm washing away", "a data tide breaking loose",
            "an ether flare shining soft", "a chain storm sealing shut",
            "a null tide fading thin", "a holo pulse calming slow",
            "a reality spark igniting all", "a byte rift mending true",
            "a grid flare fading out", "a quantum tide flowing free"
        ]
        
        self.running = True
        print(self.welcome_message)
        print(self.copyright_notice)
        print(" AVAILABLE COMMANDS: help | new_book | end_story")

    def help(self):
        print("=== HELP MENU ===")
        print(" - help: Show this guide")
        print(" - new_book: Generate a new extravagant sci-fi saga")
        print(" - end_story: Exit the story forge")
        print("================")

    def get_user_topics(self):
        print("=== INPUT REQUIRED ===")
        print(" Enter 3 themes for your story")
        topics = []
        for i in range(3):
            try:
                topic = input(f" Theme #{i+1}: ").strip()
                if not topic:
                    print(" [Alert] No input - Defaulting to 'grid flux'")
                    topic = "grid flux"
                topics.append(topic)
                print(f" [Locked] Theme #{i+1}: {topic}")
            except Exception as e:
                print(f" [Error] Input failed: {str(e)}. Defaulting to 'grid flux'")
                topics.append("grid flux")
                print(f" [Locked] Theme #{i+1}: grid flux")
        print("================")
        return topics

    def get_article(self, phrase):
        first_word = phrase.split()[0].lower()
        exceptions = ["hour", "honor", "heir", "honest"]
        vowels = ['a', 'e', 'i', 'o', 'u']
        if first_word in exceptions or first_word[0] in vowels:
            return "an"
        return "a"

    def generate_unique_title(self):
        prefixes = ["Echoes of", "Shards of", "Veil of", "Pulse of", "Whispers of", "Threads of", "Glimmer of", "Rift of", "Bloom of", "Tide of"]
        descriptors = ["the Forked", "the Quantum", "the Neon", "the Ether", "the Glitched", "the Boundless", "the Shattered", "the Null", "the Flux", "the Holo"]
        nouns = ["Grid", "Chain", "Void", "Nexus", "Spire", "Flux", "Abyss", "Veil", "Shard", "Drift"]
        return f"{random.choice(prefixes)} {random.choice(descriptors)} {random.choice(nouns)}"

    def quantum_story_algorithm(self, topics):
        print("=== QUANTUM FORGE ACTIVE ===")
        print(" Crafting your saga...")
        title = self.generate_unique_title()
        novel = f"=== NOVEL OUTPUT: Quantum WebXOS 2025 Saga v10.2 ===\n" \
                f" TITLE: {title}\n" \
                " BEGIN STORY\n" \
                "================"

        # Define the single character and technology
        archetype_hero = random.choice(self.archetypes)
        hero_full = f"{random.choice(self.first_names)} {random.choice(self.last_names)}, {self.get_article(archetype_hero)} {archetype_hero}"
        hero_name = hero_full.split(', ')[0]  # Just the name for later use
        tech = random.choice(self.technologies)
        tech_name, tech_desc = tech.split(', ', 1)

        # Single flowing narrative with archetype only in the first part
        story_elements = [
            f"In a WebXOS 2025 multiverse where {topics[0]} shaped existence, {hero_full} emerged from {random.choice(self.settings).split(', ')[0]}.",
            f"With the {tech_name}, {tech_desc}, {hero_name} faced {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{topics[1]} loomed heavy as {hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} soul, ventured forth.",
            f"Driven to {random.choice(self.motivations)}, {hero_name} navigated {random.choice(self.settings).split(', ')[0]}.",
            f"{random.choice(self.quirks)} shifted the ether as {hero_name} wielded the {tech_name}.",
            f"{topics[2]} pulsed through {random.choice(self.settings).split(', ')[0]}, guiding {hero_name}’s path.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} confronted {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"The {tech_name} flared, a beacon against {topics[0]}, as {hero_name} stood firm.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} figure, sought to {random.choice(self.motivations)}.",
            f"{random.choice(self.quirks)} twisted {topics[1]}, altering {hero_name}’s journey.",
            f"{hero_name} battled through {random.choice(self.settings).split(', ')[0]}, {topics[2]} a constant hum.",
            f"The {tech_name}, {tech_desc}, countered {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)} as {hero_name} pressed on.",
            f"{topics[0]} warped the grid, but {hero_name} defied it with the {tech_name}.",
            f"{hero_name}’s quest to {random.choice(self.motivations)} echoed through {random.choice(self.settings).split(', ')[0]}.",
            f"{random.choice(self.quirks)} broke the silence, {topics[1]} surging around {hero_name}.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} harnessed the {tech_name} against {topics[2]}.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} spirit, faced {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"The {tech_name} pulsed, bending {topics[0]} as {hero_name} forged ahead.",
            f"{random.choice(self.quirks)} reshaped {random.choice(self.settings).split(', ')[0]}, testing {hero_name}’s will.",
            f"{topics[2]} flared as {hero_name} wielded the {tech_name} in {random.choice(self.settings).split(', ')[0]}.",
            f"{hero_name} defied {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, driven by {random.choice(self.motivations)}.",
            f"{random.choice(self.quirks)} echoed through {topics[0]}, marking {hero_name}’s struggle.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} heart, stood alone.",
            f"The {tech_name}, {tech_desc}, shattered {topics[1]}’s hold as {hero_name} pressed forward.",
            f"{hero_name} roamed {random.choice(self.settings).split(', '[0])}, chasing {random.choice(self.motivations)}.",
            f"{topics[2]} twisted as {random.choice(self.quirks)}, guiding {hero_name}’s fate.",
            f"{hero_name} battled {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, the {tech_name} aglow.",
            f"{topics[0]} trembled as {hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} mind, pressed on.",
            f"{hero_name} faced {topics[1]} with the {tech_name}, {random.choice(self.quirks)} in the air.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} sought to {random.choice(self.motivations)}.",
            f"{topics[2]} surged, the {tech_name} countering {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} resolve, defied {topics[0]}.",
            f"{random.choice(self.quirks)} warped {random.choice(self.settings).split(', ')[0]}, {hero_name} unyielding.",
            f"The {tech_name} broke {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, {topics[1]} fading.",
            f"{hero_name}’s pursuit of {random.choice(self.motivations)} reshaped {topics[2]}’s flow.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} wielded the {tech_name}, {tech_desc}.",
            f"{topics[0]} pulsed as {random.choice(self.quirks)}, challenging {hero_name}’s path.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} soul, overcame {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{topics[1]} bent under the {tech_name}’s power as {hero_name} forged on.",
            f"{hero_name} ventured into {random.choice(self.settings).split(', ')[0]}, driven by {random.choice(self.motivations)}.",
            f"{random.choice(self.quirks)} twisted {topics[2]}, marking {hero_name}’s journey.",
            f"{hero_name} battled {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, the {tech_name} aglow.",
            f"{topics[0]} echoed through {random.choice(self.settings).split(', ')[0]}, {hero_name} unbroken.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} will, defied all odds.",
            f"{topics[1]} surged as {hero_name} wielded the {tech_name} in {random.choice(self.settings).split(', ')[0]}.",
            f"{random.choice(self.quirks)} altered {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, testing {hero_name}.",
            f"{hero_name}’s quest for {random.choice(self.motivations)} redefined {topics[2]}’s essence.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} faced {topics[0]} head-on.",
            f"The {tech_name}, {tech_desc}, countered {topics[1]} as {hero_name} fought on.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} spirit, defied {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{random.choice(self.quirks)} broke {topics[2]}’s rhythm, guiding {hero_name}’s steps.",
            f"{hero_name} roamed {random.choice(self.settings).split(', ')[0]}, the {tech_name} aglow with {topics[0]}.",
            f"{topics[1]} faded as {hero_name} pursued {random.choice(self.motivations)}.",
            f"The {tech_name} pulsed, shattering {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, {hero_name} victorious.",
            f"{random.choice(self.quirks)} echoed in {random.choice(self.settings).split(', ')[0]}, {topics[2]} shifting.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} heart, pressed through.",
            f"{topics[0]} warped as {hero_name} ventured through {random.choice(self.settings).split(', ')[0]}.",
            f"The {tech_name} defied {topics[1]}, {random.choice(self.motivations)} driving {hero_name} on.",
            f"{random.choice(self.quirks)} twisted {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, {hero_name} enduring.",
            f"{hero_name} reshaped {topics[2]}’s flow through {random.choice(self.settings).split(', ')[0]}.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} mind, stood tall.",
            f"{topics[0]} pulsed as {hero_name} faced {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"The {tech_name}, {tech_desc}, countered {topics[1]}, {random.choice(self.quirks)} in play.",
            f"{hero_name}’s pursuit of {random.choice(self.motivations)} broke {topics[2]}’s chains.",
            f"{random.choice(self.settings).split(', '[0])}, trembled as {hero_name} fought on.",
            f"{hero_name} wielded the {tech_name} against {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, {topics[0]} bending.",
            f"{random.choice(self.quirks)} shifted {topics[1]}, {hero_name} unbowed.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} soul, reshaped {random.choice(self.settings).split(', ')[0]}.",
            f"{topics[2]} flared, the {tech_name} guiding {hero_name} to {random.choice(self.motivations)}.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} defied {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{topics[0]} echoed as {random.choice(self.quirks)}, marking {hero_name}’s triumph.",
            f"The {tech_name} shattered {topics[1]} as {hero_name} pressed forward.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} will, pursued {random.choice(self.motivations)}.",
            f"{random.choice(self.settings).split(', ')[0]} glowed with {topics[2]} as {hero_name} stood.",
            f"{hero_name} faced {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}, the {tech_name} pulsing with {topics[0]}.",
            f"{random.choice(self.quirks)} bent {topics[1]}, {hero_name} forging ahead.",
            f"{hero_name} roamed {random.choice(self.settings).split(', ')[0]} in pursuit of {random.choice(self.motivations)}.",
            f"{topics[2]} surged as {hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} spirit, defied fate.",
            f"The {tech_name}, {tech_desc}, broke {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)} in {random.choice(self.settings).split(', ')[0]}.",
            f"{hero_name} sought {random.choice(self.motivations)}, {topics[0]} trembling.",
            f"{random.choice(self.quirks)} altered {topics[1]}, {hero_name} resolute.",
            f"In {random.choice(self.settings).split(', ')[0]}, {hero_name} wielded the {tech_name} against {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{topics[0]} flared as {hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} heart, pressed on.",
            f"The {tech_name} countered {topics[1]}, {random.choice(self.quirks)} shifting the tide.",
            f"{hero_name}’s quest for {random.choice(self.motivations)} pulsed through {random.choice(self.settings).split(', ')[0]}.",
            f"{topics[2]} hummed as {hero_name} defied {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{random.choice(self.quirks)} twisted {topics[0]}, the {tech_name} aglow.",
            f"{hero_name}, {self.get_article(random.choice(self.traits))} {random.choice(self.traits)} spirit, bent {topics[1]} to their will.",
            f"The {tech_name} flared in {random.choice(self.settings).split(', ')[0]}, {hero_name} chasing {random.choice(self.motivations)}.",
            f"{topics[2]} pulsed as {hero_name} overcame {self.get_article(random.choice(self.conflicts))} {random.choice(self.conflicts)}.",
            f"{random.choice(self.quirks)} echoed, {topics[0]} fading under {hero_name}’s resolve.",
            f"{hero_name} stood in {random.choice(self.settings).split(', ')[0]}, {topics[1]} a distant memory."
        ]
        
        # Shuffle and select 14 elements (leaving room for the unique ending)
        random.shuffle(story_elements)
        novel += " ".join(story_elements[:14])

        # Unique ending, not drawn from the shuffled elements or outcomes list
        ending = f" In the final moment, {hero_name} cast the {tech_name} into {random.choice(self.settings).split(', ')[0]}, where {topics[0]} and {topics[1]} collided with {topics[2]}. " \
                f"The multiverse shuddered, and {hero_name} vanished into a cascade of light, leaving behind a single, unbroken thread of {random.choice(self.traits)} hope that wove itself into the fabric of WebXOS 2025 forever."
        
        novel += ending + "\n================"
        
        print(" Forge complete!")
        return novel

    def new_book(self):
        topics = self.get_user_topics()
        novel = self.quantum_story_algorithm(topics)
        print(novel)

    def end_story(self):
        print("=== SHUTTING DOWN ===")
        print(" Booki v10.2 offline - WebXOS 2025 saga archived")
        self.running = False

    def run(self):
        print("=== SYSTEM ONLINE ===")
        while self.running:
            try:
                command = input(" Command: ").strip().lower()
                print("================")
                if command == "help":
                    self.help()
                elif command == "new_book":
                    self.new_book()
                elif command == "end_story":
                    self.end_story()
                else:
                    print(" [Error] Unknown command - Use 'help'")
            except Exception as e:
                print(f" [Error] Command input failed: {str(e)}. Try again.")

booki = Booki()
booki.run()
