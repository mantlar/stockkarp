import copy
import math
import os
import logging
from random import choice, randint, shuffle
import time
from websocket import WebSocket, WebSocketTimeoutException
import threading
import requests
import json
import re
from dqn import *
from bs4 import BeautifulSoup

# Initialize logger
logging.basicConfig(
    filename=f"training-{time.ctime(time.time())}.log".replace(" ",
                                                               "").replace(":", ""),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open("./config.json") as cfg:
    CONFIG = json.load(cfg)

with open("./moves.json") as cfg:
    MOVES: list = json.load(cfg)

# SHOWDOWN WEBSOCKET URL
SHOWDOWN_WS_URL = CONFIG["websocket"]["url"]
# SHOWDOWN USERNAME
USERNAME = CONFIG["login"]["username"]
# SHOWDOWN PASSWORD
PASSWORD = CONFIG["login"]["password"]

# value representing unknown state variable (i.e. yet-to-be-revealed pokemon, pp on pokemon that haven't been active yet)
VALUE_UNKNOWN = -1


class LoginException(Exception):
    def __init__(self, res):
        self.res = res
        logging.error("Login failed. Response: %s", res)

    def __str__(self):
        return repr(self.res)


class StatNormalizer:
    def __init__(self):
        # Dual caching system
        self.stat_cache = {}  # For actual stats
        self.base_stat_cache = {}  # For base stat estimations
        self.level = 100  # Default to L100 for competitive
        
        # Pre-cache common competitive ranges
        self._precache_common_stats()
        
    def _precache_common_stats(self):
        """Pre-caches min/max for common base stat ranges"""
        for base in range(30, 256, 5):  # All reasonable base stats
            self.get_min_stat(base)
            self.get_max_stat(base)
            self.get_estimated_stat_range(base)
    
    def get_estimated_stat_range(self, base_stat):
        """
        Returns (estimated_min, estimated_max) for base stats when actual stats are unknown.
        Uses common competitive spreads (assumes 31 IVs and 252 EVs in relevant stats).
        """
        key = (base_stat, 'estimated')
        if key not in self.base_stat_cache:
            # Competitive minimum (neutral nature, 0 EVs)
            min_est = math.floor((((2 * base_stat + 31) * self.level) / 100) + 5)
            
            # Competitive maximum (beneficial nature, 252 EVs)
            max_est = math.floor((((2 * base_stat + 31 + 63) * self.level) / 100) + 5) * 1.1
            
            self.base_stat_cache[key] = (min_est, math.floor(max_est))
        return self.base_stat_cache[key]
    
    def get_min_stat(self, base, level=100):
        """Minimum possible stat (0 IV/EV, negative nature)"""
        key = (base, level, 'min')
        if key not in self.stat_cache:
            self.stat_cache[key] = math.floor(math.floor(((2 * base * level) / 100) + 5) * 0.9)
        return self.stat_cache[key]
    
    def get_max_stat(self, base, level=100):
        """Maximum possible stat (31 IV/252 EV, positive nature)"""
        key = (base, level, 'max')
        if key not in self.stat_cache:
            self.stat_cache[key] = math.floor(math.floor((((2 * base + 31 + 63) * level) / 100) + 5) * 1.1)
        return self.stat_cache[key]
    
    def normalize_base_stat(self, base_stat):
        """
        Normalizes base stats to [0,1] range where:
        - 0 = minimum possible base stat (5)
        - 1 = normalization point (100)
        - >100 base stats continue scaling linearly but are capped at 2.55
        
        Args:
            base_stat: The base stat value (typically 5-255)
            
        Returns:
            Normalized value where 100 → 1.0
        """
        base_stat_min = 5    # Minimum reasonable base stat (Shedinja HP)
        base_stat_max = 255  # Maximum base stat (Blissey HP)
        normalization_point = 100  # Base stat value that maps to 1.0
        scale_factor = 0.01
        # Clamp to reasonable bounds
        clamped = max(base_stat_min, min(base_stat, base_stat_max))
        
        # Linear scaling up to normalization point
        if clamped <= normalization_point:
            return clamped * scale_factor
        
        # For stats above 100, continue linear scaling but don't treat as "better"
        return math.sqrt(min(1.0 + (clamped - normalization_point) * scale_factor, 2.55))
    
    def normalize_stat(self, base_stat, current_value, level=100):
        level = level or self.level
        if isinstance(base_stat, str) and base_stat.lower() == 'hp':
            return self.normalize_hp(base_stat, current_value, level)
        return self._normalize_general_stat(base_stat, current_value, level)
    
    def _normalize_general_stat(self, base_stat, current_value, level):
        """Handles non-HP stats"""
        min_val = self.get_min_stat(base_stat, level)
        max_val = self.get_max_stat(base_stat, level)
        clamped = max(min(current_value, max_val), min_val)
        return (clamped - min_val) / (max_val - min_val)

    def normalize_hp(self, base_hp, current_hp, level=100):
        """Special handling for HP stat"""
        min_hp = math.floor(((2 * base_hp * level) / 100) + level + 10)
        max_hp = math.floor((((2 * base_hp + 31 + 63) * level) / 100) + level + 10)
        clamped = max(min(current_hp, max_hp), min_hp)
        return (clamped - min_hp) / (max_hp - min_hp)

class ShowdownPokemon:
    def __init__(self, pokemon_data=None):
        self.ident = VALUE_UNKNOWN
        self.details = VALUE_UNKNOWN
        self.species = VALUE_UNKNOWN
        self.level = VALUE_UNKNOWN
        self.gender = VALUE_UNKNOWN
        self.raw_data = pokemon_data if pokemon_data else {}
        self.stats = {"atk": VALUE_UNKNOWN, "def": VALUE_UNKNOWN, "spa": VALUE_UNKNOWN,
                      "spd": VALUE_UNKNOWN, "spe": VALUE_UNKNOWN}  # HP, Atk, Def, SpA, SpD, Spe
        self.statBoosts = {"atk": 6, "def": 6, "spa": 6,
                           "spd": 6, "spe": 6, "accuracy": 6, "evasion": 6}  # HP, Atk, Def, SpA, SpD, Spd
        self.hp_percentage = VALUE_UNKNOWN
        self.statusCondition = VALUE_UNKNOWN  # 'brn', 'psn', etc.
        self.moves = [{
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "type": VALUE_UNKNOWN,
            "priority": VALUE_UNKNOWN,
            "power": VALUE_UNKNOWN,
            "accuracy": VALUE_UNKNOWN,
            "disabled": False
        }, {
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "type": VALUE_UNKNOWN,
            "priority": VALUE_UNKNOWN,
            "power": VALUE_UNKNOWN,
            "accuracy": VALUE_UNKNOWN,
            "disabled": False
        }, {
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "type": VALUE_UNKNOWN,
            "priority": VALUE_UNKNOWN,
            "power": VALUE_UNKNOWN,
            "accuracy": VALUE_UNKNOWN,
            "disabled": False
        }, {
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "type": VALUE_UNKNOWN,
            "priority": VALUE_UNKNOWN,
            "power": VALUE_UNKNOWN,
            "accuracy": VALUE_UNKNOWN,
            "disabled": False
        }]  # List of move dictionaries
        self.ability = VALUE_UNKNOWN
        self.baseAbility = VALUE_UNKNOWN
        self.item = VALUE_UNKNOWN
        self.is_active = False
        self.bst = VALUE_UNKNOWN
        self.base_stats = {"atk": VALUE_UNKNOWN, "def": VALUE_UNKNOWN, "spa": VALUE_UNKNOWN,
                           "spd": VALUE_UNKNOWN, "spe": VALUE_UNKNOWN}
        self.potential_abilities = []
        self.type = VALUE_UNKNOWN
        self.type2 = VALUE_UNKNOWN
        if pokemon_data:
            self.update_from_data(pokemon_data)

    def update_from_data(self, pokemon_data):
        if 'ident' in pokemon_data:
            # strip "p1:" or "p2: "
            self.ident = pokemon_data["ident"]
        if 'details' in pokemon_data:
            details = pokemon_data['details']
            self.details = details
            # Regular expression to parse the details string
            # Groups: 1 - species, 2 - level, 3 - gender
            match = re.match(
                r'^(.+?)(?:,\s*L(\d+))?(?:,\s*([MF]))?(?:,\s*(shiny))?$', details)
            if match:
                self.species = match.group(1).strip()
                self.level = int(match.group(2)) if match.group(2) else 100
                self.gender = match.group(3) if match.group(3) else '-'
                # Check for shiny status
                self.is_shiny = True if match.group(4) else False
            else:
                # If the format is not recognized, default to the full details string as species
                self.species = details
                self.level = 100
                self.gender = '-'
                self.is_shiny = False
        if 'condition' in pokemon_data:
            conditionSplit = pokemon_data["condition"].split()
            # check for status condition
            if len(conditionSplit) > 1:
                self.statusCondition = conditionSplit[1]
            else:
                self.statusCondition = None
            hpSplit = conditionSplit[0].split('/')
            if len(hpSplit) > 1:
                self.hp_percentage = int(hpSplit[0]) / int(hpSplit[1])
            else:
                # condition is 0 fnt
                self.hp_percentage = 0
        if 'active' in pokemon_data and pokemon_data['active']:
            self.is_active = True
        if 'moves' in pokemon_data:
            # moves from the side object are only id's, the real move objects are only revealed upon
            # the first time we get valid move ids, we make new move objects and replace the value unknown ones
            self.moves = [x for x in self.moves if x["id"] != VALUE_UNKNOWN]
            new_pkmn_moves = []
            for move in pokemon_data['moves']:
                if not move in [x["id"] for x in self.moves]:
                    new_pkmn_moves.append(
                        {"move": "unknown", "id": move, "pp": -1, "maxpp": -1, "target": -1, "disabled": -1})
            self.moves.extend(new_pkmn_moves)
            for move in self.moves:
                move_lookup = [x for x in MOVES if x["name"] == move["id"]]
                if len(move_lookup) > 0:
                    move["maxpp"] = math.floor(
                        move_lookup[0]["data"]["pp"] * 1.6)
                    move["type"] = move_lookup[0]["data"]["type"]
                    # yes technically vital throw will always have an unknown priority because of this blah blah blah
                    move["priority"] = move_lookup[0]["data"]["priority"]
                    move["power"] = move_lookup[0]["data"]["power"]
                    move["accuracy"] = move_lookup[0]["data"]["accuracy"]
        if "stats" in pokemon_data:
            self.stats = pokemon_data["stats"]
        if 'baseAbility' in pokemon_data:
            self.baseAbility = pokemon_data['baseAbility']
        if 'ability' in pokemon_data:
            self.ability = pokemon_data['ability']
        if 'item' in pokemon_data:
            self.item = pokemon_data['item']

class BattleState(object):
    def __init__(self, battle):
        self.battle: ShowdownBattle = battle
        
        # Player's side
        self._playerSide: list[ShowdownPokemon] = []
        # Opponent's side
        self._opposingSide: list[ShowdownPokemon] = []
        # Active Pokémon
        self._activePlayerPokemon: ShowdownPokemon = None
        self._activeOpponentPokemon: ShowdownPokemon = None
        
        # Weather and terrain
        self.weather = "none"
        self.terrain = None
        
        # Field effects
        self.is_grassy_terrain = False
        self.is_misty_terrain = False
        self.is_electric_terrain = False
        self.is_psychic_terrain = False
        
        # Entry hazards
        self.player_entry_hazards = {
            'stealth_rock': 0,
            'spikes': 0,
            'toxic_spikes': 0,
            'sticky_web': 0
        }
        self.opponent_entry_hazards = {
            'stealth_rock': 0,
            'spikes': 0,
            'toxic_spikes': 0,
            'sticky_web': 0
        }

    @classmethod
    def from_existing(cls, existing_state):
        """
        Creates a new BattleState instance as a copy of an existing state.
        
        Args:
            existing_state: The BattleState instance to copy from
            
        Returns:
            A new BattleState instance with copied state data
        """
        new_state = cls(existing_state.battle)
        
        # Copy player's side
        new_state._playerSide = [copy.deepcopy(p) for p in existing_state._playerSide]
        # Copy opposing side
        new_state._opposingSide = [copy.deepcopy(p) for p in existing_state._opposingSide]
        
        # Set active Pokémon by finding matching identities
        new_state._activePlayerPokemon = next(
            (p for p in new_state._playerSide if p.ident == existing_state._activePlayerPokemon.ident),
            None
        )
        new_state._activeOpponentPokemon = next(
            (p for p in new_state._opposingSide if p.ident == existing_state._activeOpponentPokemon.ident),
            None
        )
        
        # Copy weather and terrain state
        new_state.weather = existing_state.weather
        new_state.terrain = existing_state.terrain
        
        # Copy field effects
        new_state.is_grassy_terrain = existing_state.is_grassy_terrain
        new_state.is_misty_terrain = existing_state.is_misty_terrain
        new_state.is_electric_terrain = existing_state.is_electric_terrain
        new_state.is_psychic_terrain = existing_state.is_psychic_terrain
        
        # Copy entry hazards
        new_state.player_entry_hazards = existing_state.player_entry_hazards.copy()
        new_state.opponent_entry_hazards = existing_state.opponent_entry_hazards.copy()
        
        return new_state

    def _get_ability_index(self, ability_name):
        if ability_name == VALUE_UNKNOWN:
            return 0
            
        ability_map = {
            # Top 50 most common abilities (VGC 2023 + Smogon OU)
            'intimidate': 1, 'speedboost': 2, 'protosynthesis': 3, 'quarkdrive': 4,
            'unburden': 5, 'levitate': 6, 'prankster': 7, 'moldbreaker': 8,
            'technician': 9, 'serenegrace': 10, 'grimneigh': 11, 'goodasgold': 12,
            'wellbakedbody': 13, 'windrider': 14, 'electromorphosis': 15,
            'thermalexchange': 16, 'supremeoverlord': 17, 'orichalcumpulse': 18,
            'hadronengine': 19, 'opportunist': 20, 'regenerator': 21, 'toughclaws': 22,
            'hugepower': 23, 'purepower': 24, 'disguise': 25, 'wonderguard': 26,
            'multiscale': 27, 'magicguard': 28, 'noguard': 29, 'shadowtag': 30,
            'arenatrap': 31, 'moxie': 32, 'sheerforce': 33, 'adaptability': 34,
            'download': 35, 'tintedlens': 36, 'magicbounce': 37, 'voltabsorb': 38,
            'waterabsorb': 39, 'flashfire': 40, 'sapsipper': 41, 'lightningrod': 42,
            'stormdrain': 43, 'telepathy': 44, 'parentalbond': 45, 'ironfist': 46,
            'sandstream': 47, 'drizzle': 48, 'drought': 49, 'snowwarning': 50,
            
            # Additional notable abilities (51-100)
            'chlorophyll': 51, 'swiftswim': 52, 'slushrush': 53, 'sandrush': 54,
            'surgesurfer': 55, 'stamina': 56, 'berserk': 57, 'competitive': 58,
            'defiant': 59, 'justified': 60, 'weakarmor': 61, 'icescales': 62,
            'filter': 63, 'solidrock': 64, 'furcoat': 65, 'thickfat': 66,
            'fluffy': 67, 'rockypayload': 68, 'sharpness': 69, 'toxicdebris': 70,
            'seedsower': 71, 'clearbody': 72, 'cudchew': 73, 'rockypayload': 74,
            'armortail': 75, 'purifyingsalt': 76, 'eartheater': 77, 'windpower': 78,
            'lingeringaroma': 79, 'myceliummight': 80, 'mimicry': 81, 'neuroforce': 82,
            'beadsofruin': 83, 'swordofruin': 84, 'tabletsofruin': 85, 'vesselofruin': 86,
            'truant': 87, 'slowstart': 88, 'defeatist': 89, 'normalize': 90,
            'klutz': 91, 'unaware': 92, 'simple': 93, 'illuminate': 94,
            'runaway': 95, 'stall': 96, 'wimpout': 97, 'emergencyexit': 98,
            'imposter': 99, 'powerconstruct': 100
        }
        return ability_map.get(ability_name.lower().replace(" ", ""), len(ability_map)+1) # Unknown abilities go to end

    def _get_item_index(self, item_name):
        if item_name == VALUE_UNKNOWN:
            return 0
            
        item_map = {
            # Top 50 most common items
            'leftovers': 1, 'heavydutyboots': 2, 'choicescarf': 3, 'choiceband': 4,
            'choicespecs': 5, 'lifeorb': 6, 'focus sash': 7, 'assaultvest': 8,
            'weaknesspolicy': 9, 'airballoon': 10, 'rockyhelmet': 11, 'safetygoggles': 12,
            'blacksludge': 13, 'flameorb': 14, 'toxicorb': 15, 'lightclay': 16,
            'mentalherb': 17, 'whiteherb': 18, 'redcard': 19, 'ejectbutton': 20,
            'ejectpack': 21, 'roomservice': 22, 'blunderpolicy': 23, 'throatspray': 24,
            'adrenalineorb': 25, 'terashard': 26, 'expertbelt': 27, 'muscleband': 28,
            'wiseglasses': 29, 'zoomlens': 30, 'brightpowder': 31, 'quickclaw': 32,
            'kingsrock': 33, 'razorclaw': 34, 'razorfang': 35, 'laggingtail': 36,
            'ironball': 37, 'stickybarb': 38, 'shedshell': 39, 'bigroot': 40,
            'bindingband': 41, 'protectivepads': 42, 'loadeddice': 43, 'covertcloak': 44,
            'boosterenergy': 45, 'mirrorherb': 46, 'punchingglove': 47, 'clearamulet': 48,
            'abilityshield': 49, 'mistyseed': 50,
            
            # Additional notable items (51-100)
            'electricseed': 51, 'grassyseed': 52, 'psychicseed': 53, 'icestone': 54,
            'berrysweet': 55, 'lovesweet': 56, 'cloversweet': 57, 'flowersweet': 58,
            'starsweet': 59, 'ribbonsweet': 60, 'eviolite': 61, 'oranberry': 62,
            'sitrusberry': 63, 'aguavberry': 64, 'figyberry': 65, 'iapapaberry': 66,
            'wikiberry': 67, 'magoberry': 68, 'lumberry': 69, 'persimberry': 70,
            'lumberry': 71, 'chopleberry': 72, 'occa berry': 73, 'passhoberry': 74,
            'wacanberry': 75, 'rindoberry': 76, 'yacheberry': 77, 'chilanberry': 78,
            'kebiaberry': 79, 'shucaberry': 80, 'cobaberry': 81, 'payapaberry': 82,
            'tangaberry': 83, 'chartiberry': 84, 'kasibberry': 85, 'habanberry': 86,
            'colburberry': 87, 'babiriberry': 88, 'chilanberry': 89, 'liechiberry': 90,
            'ganlonberry': 91, 'petayaberry': 92, 'apicotberry': 93, 'salacberry': 94,
            'micleberry': 95, 'custapberry': 96, 'jabocaberry': 97, 'rowapberry': 98,
            'keeberry': 99, 'marangaberry': 100
        }
        
        return item_map.get(item_name.lower().replace(" ", ""), len(item_map)+1) # Unknown items map to 101



    def _get_condition_index(self, condition):
        condition_map = {'': 0, 'brn': 1,
                         'psn': 2, 'par': 3, 'slp': 4, 'frz': 5, 'fnt': 6}
        return condition_map.get(condition, 0)

    def _get_type_index(self, type_str):
        if not type_str == VALUE_UNKNOWN:
            type_str = type_str.lower() if (type_str) else ''
        type_map = {
            VALUE_UNKNOWN: -1,
            '': 0,
            'normal': 1,
            'fire': 2,
            'water': 3,
            'electric': 4,
            'grass': 5,
            'ice': 6,
            'fighting': 7,
            'poison': 8,
            'ground': 9,
            'flying': 10,
            'psychic': 11,
            'bug': 12,
            'rock': 13,
            'ghost': 14,
            'steel': 15,
            'dragon': 16,
            'dark': 17,
            'fairy': 18,
            'unknown': 19,
            'shadow': 20
        }
        return type_map.get(type_str, 0)

    def _get_normalized_state_vector(self) -> list[float]:
        """ where all the magic happens """
        # Initialize state sections
        state = []
        section_lengths = {}

        # Weather normalization
        def norm_weather(w): 
            weather_map = {
                'none': 0,
                'sunny': 1,
                'rainy': 2,
                'hail': 3,
                'sand': 4,
                'fog': 5
            }
            return weather_map.get(w, 0) / 5.0

        # Terrain normalization
        def norm_terrain(t): 
            terrain_map = {
                '': 0,
                'grassy': 1,
                'misty': 2,
                'electric': 3
            }
            return terrain_map.get(t, 0) / 3.0

        # Entry hazard normalization
        def norm_entry_hazard(hazard_type, max_stack):
            return min(hazard_type, max_stack) / max_stack
        def norm_type(t): return self._get_type_index(t) / 20.0
        def norm_cond(c): return VALUE_UNKNOWN if c in [None, VALUE_UNKNOWN] else self._get_condition_index(c) / 6.0
        def norm_power(p): return 0.0 if p in [None, VALUE_UNKNOWN] else min(p, 200)/200.0
        def norm_boost(b): return (b + 6) / 12.0
        def norm_stat(stat_name, stat_val, base_stat, lvl): return self.battle._normalize_stat(stat_name, stat_val, base_stat, lvl)
        def norm_base_stat(stat, lvl) : return self.battle.stat_normalizer.normalize_base_stat(stat)
        def norm_ability(a): return self._get_ability_index(a) / 100
        def norm_item(a): return self._get_item_index(a) / 100
        def norm_accuracy(a) : return a / 100

        # Section 1: Active Player Pokemon Information
        active_player = self._activePlayerPokemon
        if active_player:
            section = [
                active_player.hp_percentage,
                norm_stat("atk", active_player.stats["atk"], active_player.base_stats["atk"], active_player.level),
                norm_stat("def", active_player.stats["def"], active_player.base_stats["def"], active_player.level),
                norm_stat("spa", active_player.stats["spa"], active_player.base_stats["spa"], active_player.level),
                norm_stat("spd", active_player.stats["spd"], active_player.base_stats["spd"], active_player.level),
                norm_stat("spe", active_player.stats["spe"], active_player.base_stats["spe"], active_player.level),
                norm_boost(active_player.statBoosts["atk"]),
                norm_boost(active_player.statBoosts["def"]),
                norm_boost(active_player.statBoosts["spa"]),
                norm_boost(active_player.statBoosts["spd"]),
                norm_boost(active_player.statBoosts["spe"]),
                norm_boost(active_player.statBoosts["accuracy"]),
                norm_boost(active_player.statBoosts["evasion"]),
                norm_ability(active_player.ability) if active_player.ability != VALUE_UNKNOWN else VALUE_UNKNOWN,
                norm_item(active_player.item) if active_player.item != VALUE_UNKNOWN else VALUE_UNKNOWN,
                norm_cond(active_player.statusCondition if active_player.statusCondition else ''),
                norm_type(active_player.type),
                norm_type(active_player.type2 if active_player.type2 is not None else '')
            ]
        else:
            section = [VALUE_UNKNOWN] * 18
        state += section
        section_lengths['player'] = len(section)
        logging.info(f"Player Section Length: {section_lengths['player']}")

        # Section 2: Active Opponent Pokemon Information
        active_opponent = self._activeOpponentPokemon
        if active_opponent:
            section = [
                active_opponent.hp_percentage,
                norm_base_stat(active_opponent.base_stats["atk"], active_opponent.level),
                norm_base_stat(active_opponent.base_stats["def"], active_opponent.level),
                norm_base_stat(active_opponent.base_stats["spa"], active_opponent.level),
                norm_base_stat(active_opponent.base_stats["spd"], active_opponent.level),
                norm_base_stat(active_opponent.base_stats["spe"], active_opponent.level),
                norm_boost(active_opponent.statBoosts["atk"]),
                norm_boost(active_opponent.statBoosts["def"]),
                norm_boost(active_opponent.statBoosts["spa"]),
                norm_boost(active_opponent.statBoosts["spd"]),
                norm_boost(active_opponent.statBoosts["spe"]),
                norm_boost(active_opponent.statBoosts["accuracy"]),
                norm_boost(active_opponent.statBoosts["evasion"]),
                norm_ability(active_opponent.ability) if active_opponent.ability != VALUE_UNKNOWN else VALUE_UNKNOWN,
                norm_item(active_opponent.item) if active_opponent.item != VALUE_UNKNOWN else VALUE_UNKNOWN,
                norm_cond(active_opponent.statusCondition if active_opponent.statusCondition else ''),
                norm_type(active_opponent.type),
                norm_type(active_opponent.type2 if active_opponent.type2 is not None else '')
            ]
        else:
            section = [VALUE_UNKNOWN] * 18
        state += section
        section_lengths['opponent'] = len(section)
        logging.info(f"Opponent Section Length: {section_lengths['opponent']}")

        # Section 3: Player's Moves Information
        if active_player:
            moves_section = []
            for move in active_player.moves[:4]:
                move_info = [
                    1.0 if not move.get('disabled', False) else 0.0,
                    move.get('pp', 1) / move.get('maxpp', 1),
                    norm_type(move.get('type', '')),
                    move.get('priority', 0) / 6,
                    norm_power(move.get('power', 0)) if move.get('power', 0) != None else 0,
                    norm_accuracy(move.get('accuracy', 0)) if move.get('accuracy', 0) != None else 100
                ]
                moves_section += move_info
            # Ensure 4 moves with 6 attributes each
            while len(moves_section) < 24:
                moves_section += [VALUE_UNKNOWN] * (24 - len(moves_section))
        else:
            moves_section = [VALUE_UNKNOWN] * 24
        state += moves_section
        section_lengths['moves'] = len(moves_section)
        logging.info(f"Moves Section Length: {section_lengths['moves']}")

        # Section 4: Player's Team Overview
        team_section = []
        for pokemon in self._opposingSide[:6]:
            pokemon_info = [
                pokemon.hp_percentage,
                norm_type(pokemon.type),
                norm_type(pokemon.type2 if pokemon.type2 is not None else ''),
                norm_type(pokemon.moves[0].get('type', '')) if len(pokemon.moves) > 0 else VALUE_UNKNOWN,
                norm_type(pokemon.moves[1].get('type', '')) if len(pokemon.moves) > 1 else VALUE_UNKNOWN,
                norm_type(pokemon.moves[2].get('type', '')) if len(pokemon.moves) > 2 else VALUE_UNKNOWN,
                norm_type(pokemon.moves[3].get('type', '')) if len(pokemon.moves) > 3 else VALUE_UNKNOWN,
            ]
            team_section += pokemon_info
        # Pad with unknown if less than 6
        while len(team_section) < 42:  # 7 attributes per Pokémon x 6 pokemon
            team_section += [VALUE_UNKNOWN] * (42 - len(team_section))
        state += team_section
        section_lengths['player_team'] = len(team_section)
        logging.info(f"Player Team Section Length: {section_lengths['player_team']}")

        # Section 5: Opponent's Team Overview
        opponent_team_section = []
        for pokemon in self._opposingSide[:6]:
            pokemon_info = [
                pokemon.hp_percentage,
                norm_type(pokemon.type),
                norm_type(pokemon.type2 if pokemon.type2 is not None else ''),
                norm_type(pokemon.moves[0].get('type', '')) if len(pokemon.moves) > 0 else VALUE_UNKNOWN,
                norm_type(pokemon.moves[1].get('type', '')) if len(pokemon.moves) > 1 else VALUE_UNKNOWN,
                norm_type(pokemon.moves[2].get('type', '')) if len(pokemon.moves) > 2 else VALUE_UNKNOWN,
                norm_type(pokemon.moves[3].get('type', '')) if len(pokemon.moves) > 3 else VALUE_UNKNOWN,
            ]
            opponent_team_section += pokemon_info
        # Pad with unknown if less than 6
        while len(opponent_team_section) < 42:  # 7 attributes per Pokémon
            opponent_team_section += [VALUE_UNKNOWN] * (42 - len(opponent_team_section))
        state += opponent_team_section
        section_lengths['opponent_team'] = len(opponent_team_section)
        logging.info(f"Opponent Team Section Length: {section_lengths['opponent_team']}")

        # Section 6: Weather and Terrain
        weather_section = [
            norm_weather(self.weather),
            1 if self.is_grassy_terrain else 0,
            1 if self.is_misty_terrain else 0,
            1 if self.is_electric_terrain else 0,
            1 if self.is_psychic_terrain else 0
        ]
        state += weather_section
        section_lengths['weather'] = len(weather_section)
        logging.info(f"Weather Section Length: {section_lengths['weather']}")

        # Section 7: Player Entry Hazards
        player_hazard_section = [
            norm_entry_hazard(self.player_entry_hazards['stealth_rock'], 1),
            norm_entry_hazard(self.player_entry_hazards['spikes'], 3),
            norm_entry_hazard(self.player_entry_hazards['toxic_spikes'], 2),
            norm_entry_hazard(self.player_entry_hazards['sticky_web'], 1)
        ]
        state += player_hazard_section
        section_lengths['player_hazards'] = len(player_hazard_section)
        logging.info(f"Player Hazards Section Length: {section_lengths['player_hazards']}")

        # Section 8: Opponent Entry Hazards
        opponent_hazard_section = [
            norm_entry_hazard(self.opponent_entry_hazards['stealth_rock'], 1),
            norm_entry_hazard(self.opponent_entry_hazards['spikes'], 3),
            norm_entry_hazard(self.opponent_entry_hazards['toxic_spikes'], 2),
            norm_entry_hazard(self.opponent_entry_hazards['sticky_web'], 1)
        ]
        state += opponent_hazard_section
        section_lengths['opponent_hazards'] = len(opponent_hazard_section)
        logging.info(f"Opponent Hazards Section Length: {section_lengths['opponent_hazards']}")

        # Log total state length
        logging.info(f"Total State Length: {len(state)}")
        return state

class ShowdownBattle(object):
    """ Showdown battle context """

    def __init__(self, connection):
        self._connection: ShowdownConnection = connection
        self._roomId = 0
        self._roomName = ""
        self._format = ""
        self._p1Name = ""
        self._p2Name = ""
        self._playerNumber = -1
        self.current_state : BattleState | None = None
        self.last_state  : BattleState | None = None
        self.last_state_vector: list[int] | None = None
        self.last_action = 0
        self.last_valid_actions : list[bool] | None = None
        self.last_request = None
        self.waiting_for_details = False
        self.stat_normalizer = StatNormalizer()


    def update_player_side(self, side_data):
        for pokemon_data in side_data['pokemon']:
            existing_pokemon = next((p for p in self.current_state._playerSide if p.raw_data.get(
                'ident', '') == pokemon_data.get('ident', '')), None)
            if existing_pokemon:
                existing_pokemon.update_from_data(pokemon_data)
            else:
                new_pokemon = ShowdownPokemon(pokemon_data=pokemon_data)
                self._connection.sendCommand(f"/details {new_pokemon.species}")
                time.sleep(1)       # TODO this is profoundly fucking lame but if it's commented out it randomly breaks handling requests
                self.current_state._playerSide.append(new_pokemon)
                if new_pokemon.is_active:
                    self.current_state._activePlayerPokemon = new_pokemon

    def update_active_pokemon(self, active_data):
        """ Parse the active object in the request object. """
        for i, v in enumerate(self.current_state._activePlayerPokemon.moves):
            # TODO: the second turn of moves like outrage that lock a poke into a move are not the same as the actual move, and this causes a move to be overwritten by outrage (see training-FriJun270103262025.log)
            if active_data[0]["moves"][i]["id"] in [x["id"] for x in self.current_state._activePlayerPokemon.moves]:
                v.update(active_data[0]["moves"][i])

    def update_opposing_side(self, opposing_data):
        for pokemon_data in opposing_data['pokemon']:
            existing_pokemon = next((p for p in self.current_state._opposingSide if p.raw_data.get(
                'ident', '') == pokemon_data.get('ident', '')), None)
            if existing_pokemon:
                existing_pokemon.update_from_data(pokemon_data)
            else:
                new_pokemon = ShowdownPokemon(pokemon_data)
                self.current_state._opposingSide.append(new_pokemon)
                if new_pokemon.is_active:
                    self._activeOpponentPokemon = new_pokemon

    def _normalize_stat(self, stat_name, current_value, base_stat, level):
        """Wrapper for stat normalization"""
        if stat_name == 'hp':
            return self.stat_normalizer.normalize_hp(base_stat, current_value, level)
        return self.stat_normalizer.normalize_stat(base_stat, current_value, level)
    
    def _get_state_vector(self):
        return self.current_state._get_normalized_state_vector()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"[BATTLE] {self._roomId} : {self._p1Name} vs {self._p2Name}; [ACTIVE] {self._activePlayerPokemon}; [SIDE] {self.current_state._playerSide}"


class ShowdownConnection(object):
    """ Class that represents a showdown session. Start session using loginToServer """

    def __init__(self, username, password, useTeams=None, timeout=1):
        self.webSocket = WebSocket()
        self.webSocket.settimeout(CONFIG["websocket"]["timeout"])
        self.recvThread = threading.Thread(
            target=self.loopRecv, name="recvThread", args=(), daemon=True)
        self.detailParserThread = threading.Thread(target=self.parseDetails, name="detailParserThread", args=(), daemon=True)
        self.username = username
        self.password = password
        self.loggedIn = False
        self._lock = threading.Lock()
        self._exit = False
        self._currentBattles: dict[str, ShowdownBattle] = {}
        self._messageHandlers = {
            "updateuser": self.handleNotImplemented,
            "updatesearch": self.handleUpdateSearch,
            "challstr": self.handleChallStr,
            "pm": self.handlePm,
            ">": self.handleRoomUpdate
        }
        if useTeams == None:
            self.useTeams = {}
        else:
            self.useTeams = useTeams
        # 13 features, 9 actions
        self.agent = DQNAgent(state_size=157, action_size=9)
        self.batch_size = 32
        self.last_battle_id = None
        logging.info(
            "Initialized ShowdownConnection for username: %s", username)

    def _get_pokemon_features(self, pokemon: ShowdownPokemon, is_opponent=False) -> list[float]:
        features = []

        # Type IDs (Convert to numerical values)
        type_id = self.type_to_index(pokemon['type'])
        features.append(type_id)

        # Move IDs (Hash move IDs)
        for move in pokemon['moves'][:4]:
            move_id = self._get_hash(move['id'])
            features.append(move_id)

        # Current and Max HP
        hp_ratio = pokemon['condition']['curr'] / pokemon['condition']['max']
        features.append(hp_ratio)

        # Status Condition (Convert to numerical value)
        status_id = self._get_condition_index(pokemon['condition']['status'])
        features.append(status_id)

        return features

    def _get_valid_actions_mask(self, reqObject):
        """Create mask of valid actions (moves + switches)"""
        # Initialize with all actions invalid
        valid_actions = [False] * 9  # 4 moves + 5 switches

        # Handle moves
        if not reqObject.get('forceSwitch', [False])[0]:
            if 'active' in reqObject and reqObject['active']:
                for i, move in enumerate(reqObject['active'][0]['moves'][:4]):
                    valid_actions[i] = not move.get('disabled', False)

        # Handle switches
        available_switches = []
        for i, p in enumerate(reqObject['side']['pokemon']):
            if not p['active'] and 'fnt' not in p['condition']:
                available_switches.append(i)

        # Enable switch actions for available Pokémon
        for j in range(min(5, len(available_switches))):
            valid_actions[4 + j] = True

        return valid_actions

    def loginToServer(self, url=SHOWDOWN_WS_URL):
        """ Connects self.webSocket to the showdown server at url. """
        if not self.loggedIn:
            self.webSocket.connect(url)
            logging.info("Connected to server: %s", url)

    def sendCommand(self, command, room_id=None):
        """ Sends the given command, starting with "/", to the given room, or global if not specified. """
        if room_id == None:
            room_id = ""
        # in case i forget to add a slash
        if command[0] != "/":
            command = "/" + command
        self.webSocket.send(room_id + "|" + command)
        logging.info("Sent command to server: %s", command)

    def loopRecv(self):
        while not self._exit:
            pl = self.recvAllFromServer()
            if len(pl) > 0:
                for msg in pl:
                    # do shit idk what yet
                    if msg[0] == "|":
                        msg = msg[1:].split("|")
                    for h in self._messageHandlers:
                        if msg[0].startswith(h) and self._messageHandlers[h](msg):
                            # logging.info("Handled message: %s", msg)
                            break
            time.sleep(CONFIG["loopSleep"])

    def parseDetails(self):
        while not self._exit:
            for id, battle in self._currentBattles.items():
                if battle.waiting_for_details:
                    if battle.current_state._activePlayerPokemon.type != VALUE_UNKNOWN and battle.current_state._activePlayerPokemon.type != VALUE_UNKNOWN:
                        self.decideAction(
                            reqObject=battle.last_request, roomName=battle._roomName, battle=battle)
                        battle.waiting_for_details = False
            time.sleep(CONFIG["loopSleep"])

    def handleRequest(self, args, roomid):
        capture = True
        req_object = json.loads(args[1])
        if not 'wait' in req_object:
            logging.info("Handling request for battle: %s", roomid)
            battle = self._currentBattles.get(roomid)
            if not battle.current_state:
                if battle.last_state:
                    battle.current_state = BattleState.from_existing(battle.last_state)
                else:
                    battle.current_state = BattleState(battle)
            if battle:
                battle.last_request = req_object
                if 'side' in req_object:
                    battle.update_player_side(req_object['side'])
                if 'active' in req_object:
                    battle.update_active_pokemon(req_object["active"])
                if 'opposingSide' in req_object:
                    battle.update_opposing_side(req_object['opposingSide'])

                player_pokemon = battle.current_state._activePlayerPokemon
                opponent_pokemon = battle.current_state._activePlayerPokemon

                if player_pokemon and player_pokemon.type != VALUE_UNKNOWN and opponent_pokemon and opponent_pokemon.type != VALUE_UNKNOWN:
                    self.decideAction(reqObject=json.loads(
                        args[1]), roomName=roomid, battle=battle)
                # we need to wait for details to come back
                elif battle.current_state._activePlayerPokemon and battle.current_state._activePlayerPokemon and (battle.current_state._activePlayerPokemon.type == VALUE_UNKNOWN or battle.current_state._activePlayerPokemon.type == VALUE_UNKNOWN):
                    battle.waiting_for_details = True
        return capture

    def handleTurnEvents(self, lines, roomid):
        capture = True
        battle = self._currentBattles.get(roomid)
        if battle:
            if not battle.current_state:
                if battle.last_state:
                    battle.current_state = BattleState.from_existing(battle.last_state)
                else:
                    battle.current_state = BattleState(battle)
            playerNumber = battle._playerNumber
            for line in lines:
                if line.startswith("|"):
                    line = line[1:]
                linesplit = line.split("|")
                event = linesplit[0]
                if event == "detailschange":
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    details = parts[2]
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.details = details
                if event == "switch":
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    details = parts[2]
                    hp_ratio = ''.join(
                        [x for x in parts[3] if x.isnumeric() or x == "/"])
                    currHp, maxHp = tuple(hp_ratio.split("/"))
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        if targetSide == battle.current_state._playerSide:
                            battle.current_state._activePlayerPokemon.statBoosts = {"atk": 6, "def": 6, "spa": 6,
                                                                      "spd": 6, "spe": 6, "accuracy": 6, "evasion": 6}
                            battle.current_state._activePlayerPokemon = pokemon
                        else:
                            battle.current_state._activeOpponentPokemon = pokemon
                        pokemon.hp_percentage = float(currHp) / float(maxHp)
                        battle.current_state._activeOpponentPokemon.statBoosts = {"atk": 6, "def": 6, "spa": 6,
                                                                    "spd": 6, "spe": 6, "accuracy": 6, "evasion": 6}
                    # we don't *really* care about updating the player's active pokemon here since the request will do it anyway
                    # but we do need to have something in the active opponent side for decision making
                    elif targetSide == battle.current_state._opposingSide:
                        newPoke = ShowdownPokemon()
                        newPoke.ident = pokemon_name
                        newPoke.details = details
                        match = re.match(
                            r'^(.+?)(?:,\s*L(\d+))?(?:,\s*([MF]))?(?:,\s*(shiny))?$', details)
                        if match:
                            newPoke.species = match.group(1).strip()
                            newPoke.level = int(match.group(
                                2)) if match.group(2) else 100
                            newPoke.gender = match.group(
                                3) if match.group(3) else '-'
                            # Check for shiny status
                            newPoke.is_shiny = True if match.group(
                                4) else False
                        else:
                            # If the format is not recognized, default to the full details string as species
                            newPoke.species = details
                            newPoke.level = 100
                            newPoke.gender = '-'
                            newPoke.is_shiny = False
                        if newPoke.species != -1:
                            self.sendCommand(f"/details {newPoke.species}")
                        time.sleep(1)
                        newPoke.hp_percentage = float(currHp) / float(maxHp)
                        conditionSplit = parts[3].split()
                        # check for status condition
                        if len(conditionSplit) > 1:
                            newPoke.statusCondition = conditionSplit[1]
                        else:
                            newPoke.statusCondition = None
                        battle.current_state._activeOpponentPokemon = newPoke
                        battle.current_state._opposingSide.append(newPoke)

                elif event == "move":
                    # TODO pp drain and move targeting type discovery (not super relevant for singles but might indicate a boosting move)
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    moveid = parts[2].lower().replace(" ", "").replace("-", "")
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        if targetSide == battle.current_state._opposingSide:
                            unknownmoves = [
                                x for x in pokemon.moves if x["id"] == VALUE_UNKNOWN]
                            knownmoveids = [
                                x["id"] for x in pokemon.moves if x["id"] != VALUE_UNKNOWN]
                            if len(unknownmoves) > 0 and not moveid in knownmoveids and not moveid == "struggle":
                                unknownmoves[0]["id"] = moveid
                                unknownmoves[0]["move"] = parts[2]
                                move_lookup = [
                                    x for x in MOVES if x["name"] == moveid]
                                if len(move_lookup) > 0:
                                    unknownmoves[0]["pp"] = math.floor(
                                        move_lookup[0]["data"]["pp"] * 1.6)
                                    unknownmoves[0]["maxpp"] = math.floor(
                                        move_lookup[0]["data"]["pp"] * 1.6)
                                    unknownmoves[0]["type"] = move_lookup[0]["data"]["type"]
                                    unknownmoves[0]["priority"] = move_lookup[0]["data"]["priority"]
                                    unknownmoves[0]["power"] = move_lookup[0]["data"]["power"]
                                    unknownmoves[0]["accuracy"] = move_lookup[0]["data"]["accuracy"]
                                # self.sendCommand(f"/details {moveid}")
                                # time.sleep(1)
                elif event == "-damage":
                    # Handle damage events
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    # we don't care about status for damage
                    damage_details = ''.join(
                        [x for x in parts[2] if x.isnumeric() or x == "/"])
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        if damage_details == "0":
                            pokemon.hp_percentage = 0
                            pokemon.is_active = False
                        else:
                            currHp, maxHp = tuple(damage_details.split("/"))
                            pokemon.hp_percentage = float(
                                currHp) / float(maxHp)
                elif event == "-heal":
                    # Handle healing events
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    heal_amount = ''.join(
                        [x for x in parts[2] if x.isnumeric() or x == "/"])
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        currHp, maxHp = tuple(heal_amount.split("/"))
                        pokemon.hp_percentage = float(currHp) / float(maxHp)
                elif event == "-status":
                    # Handle status conditions (e.g., burn, poison)
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    status = parts[2]
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.statusCondition = status
                elif event == "faint":
                    # Handle fainting Pokémon
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.hp_percentage = 0
                        pokemon.is_active = False
                elif event == "-boost":
                    # Handle stat boosts
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    stat = parts[2]
                    boost = parts[3]
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        current_boost = pokemon.statBoosts[stat]
                        stat_boost = {
                            stat: min(12, current_boost + int(boost))}
                        pokemon.statBoosts.update(stat_boost)
                elif event == "-unboost":
                    # Handle stat unbboosts
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    stat = parts[2]
                    boost = parts[3]
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        current_boost = pokemon.statBoosts[stat]
                        stat_boost = {stat: max(0, current_boost - int(boost))}
                        pokemon.statBoosts.update(stat_boost)
                elif event == "-ability":
                    # |-ability|p1a: Dialga|Pressure
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    abilityname = parts[2].lower().replace(
                        " ", "").replace("-", "")
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.ability = abilityname
                elif event == "-item":
                    # |-item|p1a: Gurdurr|Choice Specs|[from] move: Trick
                    parts = linesplit
                    pokemon_name = parts[1].replace(
                        "p1a", "p1").replace("p2a", "p2")
                    item = parts[2].lower().replace(" ", "").replace("-", "")
                    targetSide = battle.current_state._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith(
                        "p2") and battle._playerNumber == 2) else battle.current_state._opposingSide
                    pokemon = next(
                        (p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.item = item
                elif event == "-fail":
                    # TODO handle move failure
                    pass
                elif event == "-sidestart":
                    side = linesplit[1]
                    move = linesplit[2]
                    if move == "Stealth Rock":
                        if side == "p1":
                            battle.player_entry_hazards['stealth_rock'] = 1
                        else:
                            battle.opponent_entry_hazards['stealth_rock'] = 1
                    elif move == "Spikes":
                        side_num = 1 if side == "p1" else 2
                        if side_num == 1:
                            battle.player_entry_hazards['spikes'] = min(battle.player_entry_hazards['spikes'] + 1, 3)
                        else:
                            battle.opponent_entry_hazards['spikes'] = min(battle.opponent_entry_hazards['spikes'] + 1, 3)
                    elif move == "Toxic Spikes":
                        side_num = 1 if side == "p1" else 2
                        if side_num == 1:
                            battle.player_entry_hazards['toxic_spikes'] = min(battle.player_entry_hazards['toxic_spikes'] + 1, 2)
                        else:
                            battle.opponent_entry_hazards['toxic_spikes'] = min(battle.opponent_entry_hazards['toxic_spikes'] + 1, 2)
                    elif move == "Sticky Web":
                        side_num = 1 if side == "p1" else 2
                        if side_num == 1:
                            battle.player_entry_hazards['sticky_web'] = 1
                        else:
                            battle.opponent_entry_hazards['sticky_web'] = 1
                    logging.info(f"Entry hazard updated: {move} for side: {side}")
                elif event == "-sideend":
                    # TODO This shit as well
                    pass
                elif event == "-weather":
                    weather = linesplit[1]
                    battle.weather = weather
                    logging.info(f"Weather updated to: {weather}")
                elif event == "-fieldstart":
                    field = linesplit[1]
                    if field == "Grassy Terrain":
                        battle.is_grassy_terrain = True
                    elif field == "Misty Terrain":
                        battle.is_misty_terrain = True
                    elif field == "Electric Terrain":
                        battle.is_electric_terrain = True
                    elif field == "Psychic Terrain":
                        battle.is_psychic_terrain = True

                    battle.terrain = field
                    logging.info(f"Terrain updated to: {field}")
                elif event == "-fieldend":
                    battle.is_grassy_terrain = False
                    battle.is_misty_terrain = False
                    battle.is_electric_terrain = False
                    battle.is_psychic_terrain = False

        return capture

    def parse_raw_data(self, html_content, battle: ShowdownBattle):
        """Parse raw HTML content and update the battle state."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Determine if it's a move or pokemon details
        if soup.find("span", class_="col movenamecol"):
            # Check for move details
            move_details = self.parse_move_details(html_content)
            if move_details:
                # Update the move in the battle
                self._update_battle_moves(battle, move_details)
                return move_details
        elif soup.find("span", "col pokemonnamecol"):
            # Otherwise, treat as pokemon details
            pokemon_details = self.parse_pokemon_details(html_content)
            if pokemon_details:
                self._update_battle_pokemon(battle, pokemon_details)
                return pokemon_details
        return None

    def _update_battle_moves(self, battle: ShowdownBattle, move_details):
        """Update the battle with the new move information."""
        # Find the pokemon that could have this move
        for pokemon in battle.current_state._playerSide:
            if any(move['id'] == move_details['id'] for move in pokemon.moves):
                # Update the move details
                for move in pokemon.moves:
                    if move['id'] == move_details['id']:
                        move['type'] = move_details['type']
                        move['category'] = move_details['category']
                        if 'power' in move:
                            move['power'] = move_details['power']
                        else:
                            move["power"] = 0
                        move['accuracy'] = move_details['accuracy']
                        move['pp'] = move_details['pp']
                        move["maxpp"] = move_details["pp"]
                        move['description'] = move_details['description']
                        # Add additional details if available
                        if 'Priority' in move_details:
                            move['priority'] = int(move_details['Priority'])
                        if 'Z-Power' in move_details:
                            move['zPower'] = int(move_details['Z-Power'])
                        if 'Target' in move_details:
                            move['target'] = move_details['Target']
                        break
        # Update opponent's moves if necessary
        for pokemon in battle.current_state._opposingSide:
            if any(move['id'] == move_details['id'] for move in pokemon.moves):
                # Update the move details (opponent's moves are not tracked, so just update the first matching move)
                for move in pokemon.moves:
                    if move['id'] == move_details['id']:
                        move['type'] = move_details['type']
                        move['category'] = move_details['category']
                        if 'power' in move:
                            move['power'] = move_details['power']
                        else:
                            move["power"] = 0
                        move['accuracy'] = move_details['accuracy']
                        move['pp'] = move_details['pp']
                        move["maxpp"] = move_details["pp"]
                        move['description'] = move_details['description']
                        if 'Priority' in move_details:
                            move['priority'] = int(move_details['Priority'])
                        if 'Z-Power' in move_details:
                            move['zPower'] = int(move_details['Z-Power'])
                        if 'Target' in move_details:
                            move['target'] = move_details['Target']
                        break

    def _update_battle_pokemon(self, battle: ShowdownBattle, new_details_obj):
        """Update the battle with the new pokemon information."""
        # Update player's pokemon
        for pokemon in battle.current_state._playerSide:
            if pokemon.species.lower() == new_details_obj['name'].lower():
                # Update base stats
                if 'base_stats' in new_details_obj:
                    pokemon.base_stats = new_details_obj['base_stats']
                # Update types
                if 'types' in new_details_obj:
                    pokemon.type = new_details_obj['types'][0]
                    if len(new_details_obj['types']) > 1:
                        pokemon.type2 = new_details_obj['types'][1]
                    else:
                        pokemon.type2 = None
                # Update abilities
                if 'abilities' in new_details_obj:
                    pokemon.potential_abilities = new_details_obj['abilities']
                # Update BST
                if 'bst' in new_details_obj:
                    pokemon.bst = new_details_obj['bst']
                # Update additional details
                if 'Dex Colour' in new_details_obj:
                    pokemon.dex_colour = new_details_obj['Dex Colour']
                if 'Egg Group(s)' in new_details_obj:
                    pokemon.egg_groups = new_details_obj['Egg Group(s)'].split(
                        ', ')
                if 'Evolution' in new_details_obj:
                    pokemon.evolution = new_details_obj['Evolution']

        # Update opponent's pokemon
        for pokemon in battle.current_state._opposingSide:
            if pokemon.species.lower() == new_details_obj['name'].lower():
                # Update base stats
                if 'base_stats' in new_details_obj:
                    pokemon.base_stats = new_details_obj['base_stats']
                # Update types
                if 'types' in new_details_obj:
                    pokemon.type = new_details_obj['types'][0]
                    if len(new_details_obj['types']) > 1:
                        pokemon.type2 = new_details_obj['types'][1]
                    else:
                        pokemon.type2 = None
                # Update abilities
                if 'abilities' in new_details_obj:
                    pokemon.potential_abilities = new_details_obj['abilities']
                # Update BST
                if 'bst' in new_details_obj:
                    pokemon.bst = new_details_obj['bst']
                # Update additional details
                if 'Dex Colour' in new_details_obj:
                    pokemon.dex_colour = new_details_obj['Dex Colour']
                if 'Egg Group(s)' in new_details_obj:
                    pokemon.egg_groups = new_details_obj['Egg Group(s)'].split(
                        ', ')
                if 'Evolution' in new_details_obj:
                    pokemon.evolution = new_details_obj['Evolution']

    def handlePm(self, args):
        capture = True
        pmFrom = args[1][1:]
        pmTo = args[2][1:]
        message = args[3]
        messageSplit = message.split()
        logging.info("Received PM command from %s: %s", pmFrom, message)
        command = messageSplit[0][1:]
        if command == "log":
            pass
        elif command == "challenge" and pmTo == self.username and pmFrom != self.username and messageSplit[0].startswith('/'):
            battleFormat = message.split()[1]
            useTeam = self.useTeams[messageSplit[1]
                                    ] if messageSplit[1] in self.useTeams else "null"
            if useTeam == None:
                useTeam = "null"
            self.sendCommand("/utm " + useTeam)
            self.sendCommand("/accept " + pmFrom)
            logging.info("Accepted challenge from %s", pmFrom)
        elif command == "raw":
            # Parse the raw data and update the current battle
            html_content = ' '.join(messageSplit[1:])
            for battleId, battle in self._currentBattles.items():
                # Update the battle with the new data
                parsed_data = self.parse_raw_data(html_content, battle)
                if parsed_data:
                    logging.info("Updated battle data for %s: %s",
                                 battleId, parsed_data)
        return capture

    def handleChallStr(self, args):
        capture = True
        challstr = args
        # challstr = challstr.split("|")
        challstr = challstr[-2] + "|" + challstr[-1]
        challRequest = requests.post("https://play.pokemonshowdown.com/api/login", data={
                                     "name": self.username, "pass": self.password, "challstr": challstr})
        # first character of challenge response is ], rest is json format
        challRequest = json.loads(challRequest.content[1:])
        if not 'assertion' in challRequest.keys():
            logging.error("Login failed. Challenge response: %s", challRequest)
            raise LoginException(challRequest)
        self.sendCommand("/trn " + self.username +
                         ",0," + challRequest['assertion'])
        self.loggedIn = True
        logging.info("Successfully logged in as %s", self.username)
        return capture

    def handleUpdateSearch(self, args):
        capture = True
        searchjson = json.loads(args[1])
        if searchjson and searchjson["games"] != None:
            for game_id in searchjson["games"]:
                # Split the game_id to extract the base room name and UUID
                parts = game_id.split('-')
                if len(parts) > 3:
                    base_room_name = '-'.join(parts[:-1])
                    uuid = parts[-1]
                    new_room_name = game_id
                else:
                    base_room_name = '-'.join(parts)
                    uuid = None
                    new_room_name = game_id

                if base_room_name not in self._currentBattles:
                    isplit = base_room_name.split("-")
                    newGame = ShowdownBattle(self)
                    newGame._format = isplit[1]
                    newGame._roomId = int(isplit[2])
                    newGame._roomName = new_room_name  # Use the new room name with UUID
                    self._currentBattles[new_room_name] = newGame
                    logging.info("New battle detected: %s", new_room_name)
                else:
                    # Update the room name in the existing battle
                    existing_battle = self._currentBattles.get(base_room_name)
                    if existing_battle:
                        existing_battle._roomName = new_room_name
                        self._currentBattles[new_room_name] = existing_battle
                        if base_room_name != new_room_name:
                            del self._currentBattles[base_room_name]
                        logging.info(
                            "Updated battle room name to: %s", new_room_name)
                        self.sendCommand("/avatar 267", room_id=new_room_name)     # based colress avatar that doesnt fucking work for some reason
            # Clean up any battles that no longer exist
            for battle_room_name in list(self._currentBattles.keys()):
                if battle_room_name not in searchjson["games"]:
                    del self._currentBattles[battle_room_name]
                    logging.info("Removed battle: %s", battle_room_name)
        elif searchjson["games"] == None:
            self._currentBattles.clear()
            logging.info("No games found. Clearing current battles.")
        return capture

    def handleRoomUpdate(self, args):
        capture = True
        roomId = ""
        lines = args.split("\n")
        for i, l in enumerate(lines):
            if l.startswith("|"):
                l = l[1:]
            lineSplit = l.split("|")
            if len(lineSplit) > 0:
                command = lineSplit[0]
                if l.startswith(">"):
                    roomId = l[1:]
                if roomId in self._currentBattles:
                    game = self._currentBattles[roomId]
                    if command == "player":
                        if lineSplit[1] == "p1":
                            game._p1Name = lineSplit[2]
                            if game._p1Name == self.username:
                                game._playerNumber = 1
                        elif lineSplit[1] == "p2":
                            game._p2Name = lineSplit[2]
                            if game._p2Name == self.username:
                                game._playerNumber = 2
                    elif command == "switch":
                        pass
                    elif command == "updateuser":
                        pass
                    elif command == "rule":
                        pass
                    elif command == "tier":
                        pass
                    elif command == "gen":
                        pass
                    elif command == "teamsize":
                        pass
                    elif command == "turn":
                        pass
                    elif command == "request":
                        self.handleRequest(lineSplit, roomId)
                    elif command == "t:":
                        self.handleTurnEvents(lines[i:], roomId)
                    elif "win" in lineSplit:
                        winner = lineSplit[-1]
                        roomId = roomId.strip()
                        battle = self._currentBattles.get(roomId)
                        if battle:
                            # Provide final reward
                            is_winner = (winner == self.username)
                            reward = 1.0 if is_winner else -1.0

                            if battle.last_state_vector is not None:
                                # Terminal state (all zeros)
                                terminal_state = [0] * len(battle.last_state_vector)
                                self.agent.remember(
                                    battle.last_state_vector, battle.last_action, reward, terminal_state, True)
                                self.agent.replay(self.batch_size)
                                torch.save(self.agent.model, "model.pth")

                            # Log the battle outcome
                            logging.info(
                                "Battle %s ended. Winner: %s", roomId, winner)
                            logging.info("Final reward: %s", reward)

                            self.sendCommand("/leave", roomId)
                            if roomId in self._currentBattles:
                                del self._currentBattles[roomId]
                    elif command == "error":
                        battle = self._currentBattles.get(roomId)
                        # Handle invalid choices
                        if "Invalid choice" in lineSplit[1]:
                            error_msg = lineSplit[1]
                            logging.error(
                                "Invalid choice error: %s", error_msg)
                            # Send undo command
                            self.sendCommand("/undo", roomId)
                            # Get the current request
                            current_request = self._currentBattles[roomId].last_request
                            if current_request:
                                # Adjust the decision-making process
                                self.adjustDecisionForTrappedPokemon(
                                    current_request, battle)
        return capture

    def adjustDecisionForTrappedPokemon(self, request, battle: ShowdownBattle):
        """
        Adjust the decision-making process when the active Pokémon is trapped.
        """
        # Get the current state and valid actions
        state = battle._get_state_vector(request)
        valid_actions = self._get_valid_actions_mask(request)

        # Remove switching options if the Pokémon is trapped
        if any("trapped" in move.get('disabledReason', '') for move in request['active'][0]['moves']):
            # Only allow move actions
            valid_actions = [True] * 4 + [False] * 5

        # Let the DQN agent choose a valid action
        action_idx = self.agent.act(state, valid_actions)

        # Store for next step
        battle.last_state = state
        battle.last_action = action_idx
        battle.last_valid_actions = valid_actions

        # Execute the action
        if action_idx < 4:  # Move
            move_id = request['active'][0]['moves'][action_idx]['id']
            command = f"move {move_id}"
        else:  # Switch (should be disabled in this case)
            command = "pass"  # or choose a default move

        # Log the command being sent
        logging.info("Executing command after adjustment: %s", command)
        self.sendCommand(command, request['room'])

    def handleNotImplemented(self, args):
        capture = False
        logging.warning("Unhandled message received: %s", args)
        return capture

    def recvAllFromServer(self):
        pl = []
        with self._lock:
            while True:
                try:
                    a = self.webSocket.recv()
                    pl += [a]
                    logging.info("Received message from server: %s", a)
                except WebSocketTimeoutException:
                    break
        if len(pl) > 0:
            self.lastMsgRecvd = pl
        return pl

    def sendChallenge(self, player, format, team=None):
        if team == None:
            team = "null"
        self.sendCommand("/utm " + team)
        self.sendCommand("/challenge " + player + ", " + format)
        logging.info("Sent challenge to player: %s", player)

    def decideAction(self, reqObject, roomName, battle: ShowdownBattle = None):
        if not battle:
            return

        # Log the start of a new decision
        logging.info("Starting decision for battle: %s", roomName)
        logging.info("Current state: %s", reqObject)

        # Store previous experience if available
        if battle.last_state_vector is not None and battle.last_valid_actions is not None:
            reward = 0  # Small intermediate reward; TODO reward function goes here
            next_state = battle._get_state_vector()
            self.agent.remember(battle.last_state_vector,
                                battle.last_action, reward, next_state, False)
            self.agent.replay(self.batch_size)

            # Log the experience replay
            logging.info("Replayed experience with reward: %s", reward)
            logging.info("Next state (len %d): %s", len(next_state), next_state)

        # Get current state and valid actions
        state = battle._get_state_vector()
        valid_actions = self._get_valid_actions_mask(reqObject)

        # Log the current state and valid actions
        logging.info("Current state vector: %s", state)
        logging.info("Valid actions: %s", valid_actions)

        # Choose action
        action_idx = self.agent.act(state, valid_actions)

        # Log the chosen action
        logging.info("Chosen action index: %s", action_idx)

        # Store for next step
        battle.last_state_vector = state
        battle.last_state = battle.current_state
        battle.current_state = None
        battle.last_action = action_idx
        battle.last_valid_actions = valid_actions

        # Execute action
        if action_idx < 4:  # Move
            move_id = reqObject['active'][0]['moves'][action_idx]['id']
            command = f"move {move_id}"
        else:  # Switch
            switch_idx = action_idx - 4
            available_switches = [i for i, p in enumerate(reqObject['side']['pokemon'])
                                  if not p['active'] and not p['condition'].endswith(' fnt')]
            if switch_idx < len(available_switches):
                pokemon_index = available_switches[switch_idx] + 1
                command = f"switch {pokemon_index}"
            else:  # Fallback to first available
                pokemon_index = available_switches[0] + 1
                command = f"switch {pokemon_index}"

        # Log the command being sent
        logging.info("Executing command: %s", command)
        self.sendCommand(command, room_id=roomName)

    def _get_hash(self, move_id):
        # Simple hash function for move IDs
        return abs(hash(move_id)) % (2 ** 32)

    def parse_move_details(self, html_content):
        move_details = {}

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract move name
        movename_tag = soup.find('span', class_='col movenamecol')
        if movename_tag:
            move_details['name'] = movename_tag.find('a').text.strip()
            move_details['id'] = move_details['name'].lower().replace(
                " ", "").replace("-", "")

        # Extract type and category
        typecol_tag = soup.find('span', class_='col typecol')
        if typecol_tag:
            imgs = typecol_tag.find_all('img')
            move_details['type'] = imgs[0]['alt']
            move_details['category'] = imgs[1]['alt']

        # Extract power
        power_tag = soup.find('span', class_='col labelcol')
        if power_tag:
            move_details['power'] = int(
                power_tag.find('br').next_sibling.strip())

        # Extract accuracy
        accuracy_tag = soup.find('span', class_='col widelabelcol')
        if accuracy_tag:
            accuracy = accuracy_tag.find(
                'br').next_sibling.strip().replace('%', '')
            if accuracy == "—":
                move_details["accuracy"] = 255
            else:
                move_details['accuracy'] = int(accuracy)

        # Extract PP
        pp_tag = soup.find('span', class_='col pplabelcol')
        if pp_tag:
            move_details['pp'] = int(pp_tag.find('br').next_sibling.strip())

        # Extract description
        desc_tag = soup.find('span', class_='col movedesccol')
        if desc_tag:
            move_details['description'] = desc_tag.text.strip()

        return move_details

    def parse_pokemon_details(self, html_content):
        pokemon_details = {}

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract main details from utilichart
        utilichart = soup.find('ul', class_='utilichart')
        if utilichart:
            # Extract Pokemon name
            name_tag = utilichart.find('span', class_='col pokemonnamecol')
            if name_tag:
                pokemon_details['name'] = name_tag.find('a').text.strip()

            # Extract types
            type_tag = utilichart.find('span', class_='col typecol')
            if type_tag:
                type_imgs = type_tag.find_all('img')
                pokemon_details['types'] = [img['alt'] for img in type_imgs]

            # Extract abilities
            abilities = []
            ability_one = utilichart.find_all('span', class_='col abilitycol')
            ability_two = utilichart.find_all(
                'span', class_='col twoabilitycol')
            if ability_one:
                abilities += [tag.text.strip()
                              for tag in ability_one if len(tag.text.strip()) > 0]
            if ability_two:
                for i in ability_two:
                    if len(i.contents) > 2:
                        abilities += [i.contents[0], i.contents[2]]
            pokemon_details['abilities'] = abilities

            # Extract base stats
            stat_tags = utilichart.find_all('span', class_='col statcol')
            if stat_tags:
                stats = {}
                for tag in stat_tags:
                    text = tag.text
                    stat_name = ''.join([x.lower()
                                        for x in text if x.isalpha()])
                    stat_val = int(''.join([x.lower()
                                   for x in text if x.isdecimal()]))
                    stats[stat_name] = stat_val
                pokemon_details['base_stats'] = stats

            # Extract BST
            bst_tag = utilichart.find('span', class_='col bstcol')
            if bst_tag:
                bst = bst_tag.find('em').text.replace('BST', '').strip()
                pokemon_details['bst'] = int(bst) if bst.isdigit() else bst

        # Extract additional details from font tag
        font_tag = soup.find('font', size="1")
        if font_tag:
            parts = font_tag.text.split('&ThickSpace;')
            for part in parts:
                if not part:
                    continue
                key_value = part.strip().split(': ')
                if len(key_value) == 2:
                    key, value = key_value
                    # Handle special cases
                    if key == 'Evolution':
                        evolution = value.strip()
                        if evolution.endswith(')'):
                            evolution = evolution.rsplit('(', 1)[0].strip()
                        pokemon_details[key] = evolution
                    else:
                        pokemon_details[key.strip()] = value.strip()

        return pokemon_details

    def Start(self, model=None):
        logging.info("%s \nStarting Stockkarp.", ("=" * 100))
        if not self.loggedIn:
            if model != None:
                self.agent.model = model
            self.loginToServer()
            self.recvThread.start()
            self.detailParserThread.start()
            logging.info(
                "Started ShowdownConnection. Username: %s", self.username)

    def Stop(self):
        if self.loggedIn:
            self._exit = True
            self.recvThread.join()
            self.detailParserThread.join()
            logging.info(
                "Stopped ShowdownConnection.")


def pokepasteToPacked(url):
    packTeam = ""
    res = requests.get(url).content.decode(
        "UTF-8").replace("\t", "").replace("\n", "")
    content = re.findall("(<pre>.*?</pre>)", res)
    for i in content:
        nick = species = item = ability = evs = ivs = nature = ""
        moves = []
        text = re.sub("<(.*?)>", "", i)
        for i, j in enumerate(text.split("  ")):
            # print(j)
            # TODO: add gender, right now it just ignores gender and assumes it will be set if there is only one legal gender, which is fine in most cases but not optimal
            j = j.replace("(M)", "").replace("(F)", "")
            if i == 0:
                hasItem = j.find("@")
                if hasItem > 0:
                    item = j[hasItem + 2:]
                    j = j[:hasItem - 1]
                nick_check = j.strip(") ").split("(")
                if len(nick_check) > 1:
                    species = nick_check[1]
                    nick = nick_check[0][:-1]
                else:
                    species = nick_check[0]
            elif j.startswith("Ability:"):
                ability = j[9:]
            elif j.startswith("EVs:"):
                evs = j[5:]
                evs = packEvs([x.strip() for x in evs.split("/")])
            elif j.startswith("IVs:"):
                ivs = j[5:]
                ivs = packEvs([x.strip() for x in ivs.split("/")])
            elif j.endswith("Nature"):
                nature = j[:-7]
            elif j.startswith("- "):
                moves += [j[2:]]
        if nick == None:
            nick = species
            species = ""
        moves = ",".join([x.lower().replace(" ", "").replace(
            "-", "").replace("[", "").replace("]", "") for x in moves])
        item = item.lower().replace(" ", "").replace("-", "")
        ability = ability.lower().replace(" ", "").replace("-", "")
        if nick:
            # print(f"{nick}|{species}|{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]")
            packTeam += f"{nick}|{species}|{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]"
        else:
            # print(f"{species}||{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]"),
            packTeam += f"{species}||{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]"
    return packTeam[:-1]  # strip final ] character


def packEvs(evList):
    evs = ["", "", "", "", "", ""]
    for i in evList:
        if i[-2:] == "HP":
            evs[0] = i[:-3]
        elif i[-3:] == "Atk":
            evs[1] = i[:-4]
        elif i[-3:] == "Def":
            evs[2] = i[:-4]
        elif i[-3:] == "SpA":
            evs[3] = i[:-4]
        elif i[-3:] == "SpD":
            evs[4] = i[:-4]
        elif i[-3:] == "Spe":
            evs[5] = i[:-4]
    logging.info("Packed EVs: %s", evs)
    return ",".join(evs)

# def get_min_stat(base, level):
#     """ Given a base stat and level, returns a stat value with 0 IVs, 0 EVs, and a negative nature. Used for stat normalization. """
#     return math.floor(math.floor(((2 * base * level) / 100) + 5) * 0.9)

# def get_max_stat(base, level):
#     """ Given a base stat and level, returns a stat value with 31 IVs and 252 EVs, and a positive nature. Used for stat normalization. """
#     return math.floor(math.floor((((2 * base + 31 + (252 / 4)) * level) / 100) + 5) * 1.1)


if __name__ == "__main__":
    try:
        os.remove("./training.log")
        print(f"File ./training.log deleted successfully.")
    except FileNotFoundError:
        print(f"File ./training.log not found.")
    except PermissionError:
        print(f"Permission denied to delete './training.log")
    except OSError as e:
        print(f"Error deleting './training.log': {e}")
    GEN9_USE_TEAM = \
        "Gliscor||toxicorb|poisonheal|poisonjab,knockoff,protect,spikes|Impish|244,,248,,16,|||||]\
        Alomomola||heavydutyboots|regenerator|scald,flipturn,wish,protect|Sassy|4,,252,,252,|||||]\
        Ting-Lu||heavydutyboots|vesselofruin|earthquake,stealthrock,whirlwind,rest|Impish|252,4,252,,,|||||]\
        Toxapex||heavydutyboots|regenerator|toxic,banefulbunker,recover,haze|Calm|252,,4,,252,|||||]\
        Blissey||heavydutyboots|naturalcure|seismictoss,soft-boiled,calmmind,protect|Calm|4,,252,,252,|||||]\
        Dondozo||heavydutyboots|unaware|avalanche,rest,sleeptalk,curse|Impish|248,,252,,8,|||||"
    teams = ["https://pokepast.es/ad7a9a738ec1a82a", "https://pokepast.es/f09ff66281cee40e",
             "https://pokepast.es/4972596fc1ed58c2", "https://pokepast.es/1214a19de0fe9ba7"]
    with open("./asciikarp.txt") as ascii:
        ascii_lines = ''.join(ascii.readlines())
        print(ascii_lines)
    logging.info(
        "Starting main script. Loading teams and initializing connection.")
    useTeams = {
        "gen9ou": GEN9_USE_TEAM,
        "gen3ou": pokepasteToPacked(choice(teams))
    }
    sd = ShowdownConnection(
        USERNAME, PASSWORD, timeout=CONFIG["websocket"]["timeout"])
    #
    # loginThread.start()
    # loginThread.join()
    model = None
    dry = CONFIG["dry"]
    if os.path.exists("./model.pth") and not dry:
        model = torch.load("model.pth")
        logging.info("Loaded existing model from model.pth")
    sd.Start(model=model)
    while not sd._exit:
        pass
        # sd.parseDetails()
