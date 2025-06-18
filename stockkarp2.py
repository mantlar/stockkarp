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

# Initialize logger
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open("./config.json") as cfg:
    CONFIG = json.load(cfg)

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
                           "spd": 6, "spe": 6}  # HP, Atk, Def, SpA, SpD, Spd
        self.hp_percentage = 1.0  # Default to full HP if not specified
        self.statusCondition = VALUE_UNKNOWN  # 'brn', 'psn', etc.
        self.moves = [{
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "disabled": False
        }, {
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "disabled": False
        }, {
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "disabled": False
        }, {
            "move": VALUE_UNKNOWN,
            "id": VALUE_UNKNOWN,
            "pp": VALUE_UNKNOWN,
            "maxpp": VALUE_UNKNOWN,
            "target": VALUE_UNKNOWN,
            "disabled": False
        }]  # List of move dictionaries
        self.ability = VALUE_UNKNOWN
        self.baseAbility = VALUE_UNKNOWN
        self.item = VALUE_UNKNOWN
        self.is_active = False
        if pokemon_data:
            self.update_from_data(pokemon_data)

    def update_from_data(self, pokemon_data):
        if 'ident' in pokemon_data:
            # strip "p1:" or "p2: "
            self.ident = pokemon_data["ident"]
        if 'details' in pokemon_data:
            self.details = pokemon_data["details"]
            detailSplit = [x.strip()
                           for x in pokemon_data["details"].split(",")]
            self.species = detailSplit[0]

            if len(detailSplit[1]) > 1 and detailSplit[1][1:].isnumeric():
                self.level = int(detailSplit[1][1:])
            else:
                self.level = 100
                
            if len(detailSplit) > 2:
                self.gender = detailSplit[2]
            elif len(detailSplit) > 1:
                self.gender = detailSplit[1]
            else:
                self.gender = "-"
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
            for i in pokemon_data['moves']:
                if not i in [x["id"] for x in self.moves]:
                    new_pkmn_moves.append(
                        {"move": "unknown", "id": i, "pp": -1, "maxpp": -1, "target": -1, "disabled": -1})
            self.moves.extend(new_pkmn_moves)
        if "stats" in pokemon_data:
            self.stats = pokemon_data["stats"]
        if 'baseAbility' in pokemon_data:
            self.baseAbility = pokemon_data['baseAbility']
        if 'ability' in pokemon_data:
            self.ability = pokemon_data['ability']
        if 'item' in pokemon_data:
            self.item = pokemon_data['item']


class ShowdownBattle(object):
    """ Showdown battle context """

    def __init__(self):
        self._roomId = 0
        self._roomName = ""
        self._format = ""
        self._p1Name = ""
        self._p2Name = ""
        self._playerNumber = -1
        # List of ShowdownPokemon instances for the player's team
        self._playerSide: list[ShowdownPokemon] = []
        # List of ShowdownPokemon instances for the opponent's team
        self._opposingSide: list[ShowdownPokemon] = []
        # Currently active Pokémon of the player
        self._activePlayerPokemon: ShowdownPokemon = None
        # Currently active Pokémon of the opponent
        self._activeOpponentPokemon: ShowdownPokemon = None
        self.last_state : list[int] | None = None
        self.last_action = 0
        self.last_valid_actions = list[bool] | None 

    def update_player_side(self, side_data):
        for pokemon_data in side_data['pokemon']:
            existing_pokemon = next((p for p in self._playerSide if p.raw_data.get(
                'ident', '') == pokemon_data.get('ident', '')), None)
            if existing_pokemon:
                existing_pokemon.update_from_data(pokemon_data)
            else:
                new_pokemon = ShowdownPokemon(pokemon_data=pokemon_data)
                self._playerSide.append(new_pokemon)
                if new_pokemon.is_active:
                    self._activePlayerPokemon = new_pokemon

    def update_active_pokemon(self, active_data):
        """ Parse the active object in the request object. """
        self._activePlayerPokemon.moves = active_data[0]["moves"]

    def update_opposing_side(self, opposing_data):
        for pokemon_data in opposing_data['pokemon']:
            existing_pokemon = next((p for p in self._opposingSide if p.raw_data.get(
                'ident', '') == pokemon_data.get('ident', '')), None)
            if existing_pokemon:
                existing_pokemon.update_from_data(pokemon_data)
            else:
                new_pokemon = ShowdownPokemon(pokemon_data)
                self._opposingSide.append(new_pokemon)
                if new_pokemon.is_active:
                    self._activeOpponentPokemon = new_pokemon

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"[BATTLE] {self._roomId} : {self._p1Name} vs {self._p2Name}; [ACTIVE] {self._activePlayerPokemon}; [SIDE] {self._playerSide}"


class ShowdownConnection(object):
    """ Class that represents a showdown session. Start session using loginToServer """

    def __init__(self, username, password, useTeams=None, timeout=1):
        self.webSocket = WebSocket()
        self.webSocket.settimeout(CONFIG["websocket"]["timeout"])
        self.webSocketThread = threading.Thread(
            target=self.loopRecv, name="loopThread", args=(), daemon=True)
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
        self.agent = DQNAgent(state_size=12, action_size=9)
        self.batch_size = 32
        self.last_battle_id = None
        logging.info(
            "Initialized ShowdownConnection for username: %s", username)

    def _get_state_vector(self, reqObject) -> list[int]:
        """Convert battle state to numerical vector"""
        # Find active Pokémon from side data
        active_pokemon = None
        for p in reqObject['side']['pokemon']:
            if p['active']:
                active_pokemon = p
                break

        # Parse active Pokémon condition
        if active_pokemon:
            cond_parts = active_pokemon['condition'].split()
            hp_part = cond_parts[0]

            if 'fnt' in active_pokemon['condition']:
                # Fainted Pokémon
                hp_ratio = 0.0
                status_idx = 0
            else:
                # Parse HP values
                if '/' in hp_part:
                    current_hp, max_hp = hp_part.split('/')
                    hp_ratio = float(current_hp) / float(max_hp)
                else:
                    hp_ratio = 1.0  # Default if parsing fails

                # Parse status condition
                status_map = {'healthy': 0, 'brn': 1,
                              'psn': 2, 'par': 3, 'slp': 4, 'frz': 5}
                status_str = cond_parts[1] if len(
                    cond_parts) > 1 else 'healthy'
                status_idx = status_map.get(status_str.split(' ')[0], 0)
        else:
            # No active Pokémon found
            hp_ratio = 0.0
            status_idx = 0

        # Move availability (1 = available, 0 = disabled)
        move_available = [0, 0, 0, 0]  # Default to all disabled
        if 'active' in reqObject and reqObject['active']:
            for i, move in enumerate(reqObject['active'][0]['moves'][:4]):
                if not move.get('disabled', False):
                    move_available[i] = 1

        # Team status (remaining HP for each Pokémon)
        team_hp = []
        for p in reqObject['side']['pokemon']:
            if 'fnt' in p['condition']:
                team_hp.append(0.0)
            else:
                cond_parts = p['condition'].split()
                hp_part = cond_parts[0]
                if '/' in hp_part:
                    current_hp, max_hp = hp_part.split('/')
                    team_hp.append(float(current_hp) / float(max_hp))
                else:
                    team_hp.append(0.0)  # Default if parsing fails

        # Pad to 6 Pokémon
        while len(team_hp) < 6:
            team_hp.append(0.0)

        # Combine all features
        return [hp_ratio, status_idx] + move_available + team_hp

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

    def handleRequest(self, args, roomid):
        capture = True
        req_object = json.loads(args[1])
        if not 'wait' in req_object:
            logging.info("Handling request for battle: %s", roomid)
            battle = self._currentBattles.get(roomid)
            if battle:
                if 'side' in req_object:
                    battle.update_player_side(req_object['side'])
                if 'active' in req_object:
                    battle.update_active_pokemon(req_object["active"])
                if 'opposingSide' in req_object:
                    battle.update_opposing_side(req_object['opposingSide'])
                self.decideAction(reqObject=json.loads(
                    args[1]), roomName=roomid)
        return capture

    def handleTurnEvents(self, lines, roomid):
        capture = True
        battle = self._currentBattles.get(roomid)
        if battle:
            playerNumber = battle._playerNumber
            for line in lines:
                if line.startswith("|"):
                    line = line[1:]
                linesplit = line.split("|")
                event = linesplit[0]
                if event == "detailschange":
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    details = parts[2]
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.details = details
                if event == "switch":
                    slot = linesplit[1][0:3]
                    name = linesplit[1][5:]
                    details = linesplit[2]
                    hp = linesplit[3]
                    isPlayerSwitch = ((playerNumber == 1 and slot == "p1a") or (playerNumber == 2 and slot == "p2a"))
                    if isPlayerSwitch:
                        switchpoke = [x for x in battle._playerSide if x.details == details]
                        if len(switchpoke) > 0:
                            battle._activePlayerPokemon = switchpoke[0]
                    else:
                        switchpoke = [x for x in battle._opposingSide if x.details == details]
                        if len(switchpoke) > 0:
                            battle._activeOpponentPokemon = switchpoke[0]
                elif event == "move":
                    # TODO pp drain and move discovery
                    pass
                elif event == "-damage":
                    # Handle damage events
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    damage_details = ''.join([x for x in parts[2] if x.isnumeric() or x == "/"])    # we don't care about status for damage
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        if damage_details == "0":
                            pokemon.hp_percentage = 0
                            pokemon.is_active = False
                        else:
                            currHp, maxHp = tuple(damage_details.split("/"))
                            pokemon.hp_percentage = float(currHp) / float(maxHp)
                elif event == "-heal":
                    # Handle healing events
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    heal_amount = ''.join([x for x in parts[2] if x.isnumeric() or x == "/"])
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        currHp, maxHp = tuple(heal_amount.split("/"))
                        pokemon.hp_percentage = float(currHp) / float(maxHp)
                elif event == "-status":
                    # Handle status conditions (e.g., burn, poison)
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    status = parts[2]
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.statusCondition = status
                elif event == "faint":
                    # Handle fainting Pokémon
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        pokemon.hp_percentage = 0
                        pokemon.is_active = False
                elif event == "-boost":
                    # Handle stat boosts
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    stat = parts[2]
                    boost = parts[3]
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        current_boost = pokemon.statBoosts[stat]
                        stat_boost = {stat: min(12, current_boost + int(boost))}
                        pokemon.statBoosts.update(stat_boost)
                elif event == "-unboost":
                    # Handle stat unbboosts
                    parts = linesplit
                    pokemon_name = parts[1].replace("p1a", "p1").replace("p2a", "p2")
                    stat = parts[2]
                    boost = parts[3]
                    targetSide = battle._playerSide if (pokemon_name.startswith("p1") and battle._playerNumber == 1 or pokemon_name.startswith("p2") and battle._playerNumber == 2) else battle._opposingSide
                    pokemon = next((p for p in targetSide if p.ident == pokemon_name), None)
                    if pokemon:
                        current_boost = pokemon.statBoosts[stat]
                        stat_boost = {stat: max(0, current_boost - int(boost))}
                        pokemon.statBoosts.update(stat_boost)
                elif event == "-fail":
                    # TODO handle move failure
                    pass
                elif event == "-sidestart":
                    # TODO more battle state shit
                    pass
                elif event == "-weather":
                    # TODO more battle state shit
                    pass
                elif event == "-fieldstart":
                    # TODO more battle state shit
                    pass
                elif event == "-fieldend":
                    # TODO more battle state shit
                    pass


        return capture

    def handlePm(self, args):
        capture = True
        pmFrom = args[1][1:]
        pmTo = args[2][1:]
        message = args[3]
        messageSplit = message.split()
        if pmTo == self.username and pmFrom != self.username and messageSplit[0].startswith('/'):
            logging.info("Received PM command from %s: %s", pmFrom, message)
            command = messageSplit[0][1:]
            if command == "log":
                pass
            elif command == "challenge":
                battleFormat = message.split()[1]
                useTeam = self.useTeams[messageSplit[1]
                                        ] if messageSplit[1] in self.useTeams else "null"
                if useTeam == None:
                    useTeam = "null"
                self.sendCommand("/utm " + useTeam)
                self.sendCommand("/accept " + pmFrom)
                logging.info("Accepted challenge from %s", pmFrom)
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
            for i in searchjson["games"]:
                if not i in self._currentBattles:
                    isplit = i.split("-")
                    newGame = ShowdownBattle()
                    newGame._format = isplit[1]
                    newGame._roomId = int(isplit[2])
                    newGame._roomName = i
                    self._currentBattles[i] = newGame
                    logging.info("New battle detected: %s", i)
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

                            if battle.last_state is not None:
                                # Terminal state (all zeros)
                                terminal_state = [0] * len(battle.last_state)
                                self.agent.remember(
                                    battle.last_state, battle.last_action, reward, terminal_state, True)
                                self.agent.replay(self.batch_size)
                                torch.save(self.agent.model, "model.pth")

                            # Log the battle outcome
                            logging.info(
                                "Battle %s ended. Winner: %s", roomId, winner)
                            logging.info("Final reward: %s", reward)

                            self.sendCommand("/leave", roomId)
                            if roomId in self._currentBattles:
                                del self._currentBattles[roomId]
        return capture

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

    def decideAction(self, reqObject, roomName):
        battle = self._currentBattles.get(roomName)
        if not battle:
            return

        # Log the start of a new decision
        logging.info("Starting decision for battle: %s", roomName)
        logging.info("Current state: %s", reqObject)

        # Store previous experience if available
        if battle.last_state is not None and battle.last_valid_actions is not None:
            reward = 0  # Small intermediate reward
            next_state = self._get_state_vector(reqObject)
            self.agent.remember(battle.last_state,
                                battle.last_action, reward, next_state, False)
            self.agent.replay(self.batch_size)

            # Log the experience replay
            logging.info("Replayed experience with reward: %s", reward)
            logging.info("Next state: %s", next_state)

        # Get current state and valid actions
        state = self._get_state_vector(reqObject)
        valid_actions = self._get_valid_actions_mask(reqObject)

        # Log the current state and valid actions
        logging.info("Current state vector: %s", state)
        logging.info("Valid actions: %s", valid_actions)

        # Choose action
        action_idx = self.agent.act(state, valid_actions)

        # Log the chosen action
        logging.info("Chosen action index: %s", action_idx)

        # Store for next step
        battle.last_state = state
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

    def type_to_index(self, type_str):
        type_map = {
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

    def _get_condition_index(self, condition):
        condition_map = {'': 0, 'brn': 1,
                         'psn': 2, 'par': 3, 'slp': 4, 'frz': 5}
        return condition_map.get(condition, 0)

    def _get_move_id_hash(self, move_id):
        # Simple hash function for move IDs
        return abs(hash(move_id)) % (2 ** 32)

    def Start(self, model=None):
        logging.info("%s \nStarting Stockkarp.", ("=" * 100))
        if not self.loggedIn:
            if model != None:
                self.agent.model = model
            self.loginToServer()
            self.webSocketThread.start()
            logging.info(
                "Started ShowdownConnection. Username: %s", self.username)

    def Stop(self):
        if self.loggedIn:
            self._exit = True
            self.webSocketThread.join()
            logging.info(
                "Stopped ShowdownConnection. Username: %s", self.username)


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
        ascii_lines = ascii.readlines()
        for i in ascii_lines:
            print(i, end='')
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
    dry = True
    if os.path.exists("./model.pth") and not dry:
        model = torch.load("model.pth")
        logging.info("Loaded existing model from model.pth")
    sd.Start(model=model)
    while not sd._exit:
        pass
