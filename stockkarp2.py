import os
from random import choice, randint, shuffle
from websocket import WebSocket, WebSocketTimeoutException
import threading
import requests
import json
import re
from dqn import *

with open("./config.json") as cfg:
    CONFIG = json.load(cfg)

# SHOWDOWN WEBSOCKET URL
SHOWDOWN_WS_URL = CONFIG["websocket"]["url"]
# SHOWDOWN USERNAME
USERNAME = CONFIG["login"]["username"]
# SHOWDOWN PASSWORD
PASSWORD = CONFIG["login"]["password"]

class LoginException(Exception):
    def __init__(self, res):
        self.res = res

    def __str__(self):
        return repr(self.res)

class ShowdownBattle(object):
    """ Showdown battle context """
    def __init__(self):
        self._roomId = 0
        self._format = ""
        self._p1Name = ""
        self._p2Name = ""
        self._playerNumber = -1
        self._active = {}
        self._side = {}
        self._opposingActive = {}
        self._opposingSide = {}
        self.last_state = None
        self.last_action = None
        self.last_valid_actions = None

    def __repr__(self) -> str:
        return self.__str__() 
    
    def __str__(self) -> str:
        return f"[BATTLE] {self._roomId} : {self._p1Name} vs {self._p2Name}; [ACTIVE] {self._active}; [SIDE] {self._side}"

    

class ShowdownConnection(object):
    """ Class that represents a showdown session. Start session using loginToServer """
    def __init__(self, username, password, useTeams=None, timeout=1):
        self.webSocket = WebSocket()
        self.webSocket.settimeout(CONFIG["websocket"]["timeout"])
        self.webSocketThread = threading.Thread(target=self.loopRecv, name="loopThread", args=(), daemon=True)
        self.username = username
        self.password = password
        self.loggedIn = False
        self._lock = threading.Lock()
        self._exit = False
        self._currentBattles : dict[str, ShowdownBattle]= {}
        self._messageHandlers = {
            "updateuser" : self.handleNotImplemented,
            "updatesearch" : self.handleUpdateSearch,
            "challstr" : self.handleChallStr,
            "pm" : self.handlePm,
            ">" : self.handleRoomUpdate
        }
        if useTeams == None:
            self.useTeams = {}
        else:
            self.useTeams = useTeams
        self.agent = DQNAgent(state_size=12, action_size=9)  # 13 features, 9 actions
        if os.path.exists("model.pth"):
            self.agent.load_model("model.pth")
        self.batch_size = 32
        self.last_battle_id = None

    def _get_state_vector(self, reqObject):
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
                status_map = {'healthy': 0, 'brn': 1, 'psn': 2, 'par': 3, 'slp': 4, 'frz': 5}
                status_str = cond_parts[1] if len(cond_parts) > 1 else 'healthy'
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

    def sendCommand(self, command, room_id=None):
        """ Sends the given command, starting with "/", to the given room, or global if not specified. """
        if room_id == None:
            room_id = ""
        # in case i forget to add a slash
        if command[0] != "/":
            command = "/" + command
        self.webSocket.send(room_id + "|" + command)

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
                            break
                        
    
    def handleRequest(self, args, roomid):
        capture = True
        req_object = json.loads(args[1])
        if not 'wait' in req_object:
            self.decideAction(req_object, roomName=roomid)
        return capture

    def handlePm(self, args):
        capture = True
        pmFrom = args[1][1:]
        pmTo = args[2][1:]
        message = args[3]
        messageSplit = message.split()
        if pmTo == self.username and pmFrom != self.username and messageSplit[0].startswith('/'):
            command = messageSplit[0][1:]
            if command == "log":
                pass
            elif command == "challenge":
                battleFormat = message.split()[1]
                useTeam = self.useTeams[messageSplit[1]] if messageSplit[1] in self.useTeams else "null"
                if useTeam == None:
                    useTeam = "null"
                self.sendCommand("/utm " + useTeam)
                self.sendCommand("/accept " + pmFrom)

        return capture


    def handleChallStr(self, args):
        capture = True
        challstr = args
        # challstr = challstr.split("|")
        challstr = challstr[-2] + "|" + challstr[-1]
        challRequest = requests.post("https://play.pokemonshowdown.com/api/login", data={"name" : self.username, "pass" : self.password, "challstr" : challstr})
        challRequest = json.loads(challRequest.content[1:])     # first character of challenge response is ], rest is json format 
        if not 'assertion' in challRequest.keys():
            raise LoginException(challRequest)
        self.sendCommand("/trn " + self.username +",0," + challRequest['assertion'])
        self.loggedIn = True
        print("Login successful! Awaiting challenge.")
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
                    self._currentBattles[i] = newGame
        elif searchjson["games"] == None:
            self._currentBattles.clear()
        return capture

    def handleRoomUpdate(self, args):
        capture = True
        lines = args.split("\n")
        for l in lines:
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
                                game._playerNumber = "p1"
                        elif lineSplit[1] == "p2":
                            game._p2Name = lineSplit[2]
                            if game._p2Name == self.username:
                                game._playerNumber = "p2"
                    elif command == "switch":
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
                                self.agent.remember(battle.last_state, battle.last_action, reward, terminal_state, True)
                                self.agent.replay(self.batch_size)
                                self.agent.save_model("model.pth")
                            
                            # Reset battle state
                            battle.last_state = None
                            battle.last_action = None
                            battle.last_valid_actions = None
                        
                        print(f"\nBattle Ended, {winner} won")
                        self.sendCommand("/leave", roomId)
                        if roomId in self._currentBattles:
                            del self._currentBattles[roomId]
        return capture

    def handleNotImplemented(self, args):
        capture = False
        return capture

    def recvAllFromServer(self):
        pl = []
        with self._lock:
            while True:
                try:
                    a = self.webSocket.recv()
                    pl += [a]
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
        

    # def decideAction(self, reqObject, roomName):
    #     if ("teamPreview" in reqObject and reqObject["teamPreview"] == True):
    #         order = ["1", "2", "3", "4", "5", "6"]
    #         shuffle(order)
    #         order = ''.join(order)
    #         print("sending team " + order)
    #         self.sendCommand("/choose team " + order, room_id=roomName)
    #     elif ("forceSwitch" in reqObject and reqObject["forceSwitch"] == [True]):
    #         switches = []
    #         for i, v in enumerate(reqObject["side"]["pokemon"]):
    #             if v["active"] == False and v["condition"] != "0 fnt":
    #                 switches.append(i + 1)
    #         c = choice(switches)
    #         print("switching to " + str(c))
    #         self.sendCommand("/choose switch " + str(c), room_id=roomName)
    #     else:
    #         if not "wait" in reqObject.keys() or reqObject["wait"] == "False":
    #             choices = []
    #             for i in reqObject["active"][0]["moves"]:
    #                 if not "disabled" in i.keys() or i["disabled"] == False:
    #                     choices.append(i["id"])
    #             c = choice(choices)
    #             print("using move " + c)
    #             self.sendCommand("/choose move " + c, room_id=roomName)

    def decideAction(self, reqObject, roomName):
        battle = self._currentBattles.get(roomName)
        if not battle:
            return
        
        # Store previous experience if available
        if battle.last_state is not None and battle.last_valid_actions is not None:
            reward = 0  # Small intermediate reward
            next_state = self._get_state_vector(reqObject)
            self.agent.remember(battle.last_state, battle.last_action, reward, next_state, False)
            self.agent.replay(self.batch_size)
        
        # Get current state and valid actions
        state = self._get_state_vector(reqObject)
        valid_actions = self._get_valid_actions_mask(reqObject)
        
        # Choose action
        action_idx = self.agent.act(state, valid_actions)
        
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
        
        print(f"Executing: {command}")
        self.sendCommand(command, room_id=roomName)

    def Start(self, model=None):
        if not self.loggedIn:
            if model != None:
                self.agent.model = model
            self.loginToServer()
            self.webSocketThread.start()

    def Stop(self):
        if self.loggedIn:
            self._exit = True
            self.webSocketThread.join()

def pokepasteToPacked(url):
    packTeam = ""
    res = requests.get(url).content.decode("UTF-8").replace("\t", "").replace("\n", "")
    content = re.findall("(<pre>.*?</pre>)", res)
    for i in content:
        nick = species = item = ability = evs = ivs = nature = ""
        moves = []
        text = re.sub("<(.*?)>", "", i)
        for i, j in enumerate(text.split("  ")):
            # print(j)
            j = j.replace("(M)", "").replace("(F)", "")     # TODO: add gender, right now it just ignores gender and assumes it will be set if there is only one legal gender, which is fine in most cases but not optimal
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
        moves = ",".join([x.lower().replace(" ", "").replace("-", "").replace("[", "").replace("]", "") for x in moves])
        item = item.lower().replace(" ", "").replace("-", "")
        ability = ability.lower().replace(" ", "").replace("-", "")
        if nick:
            # print(f"{nick}|{species}|{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]")
            packTeam += f"{nick}|{species}|{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]"
        else:
            # print(f"{species}||{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]"),
            packTeam += f"{species}||{item}|{ability.lower()}|{moves}|{nature.lower()}|{evs}||{ivs}|||]"
    return packTeam[:-1] # strip final ] character

        
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
    return ",".join(evs)

if __name__ == "__main__":
    GEN9_USE_TEAM = \
        "Gliscor||toxicorb|poisonheal|poisonjab,knockoff,protect,spikes|Impish|244,,248,,16,|||||]\
        Alomomola||heavydutyboots|regenerator|scald,flipturn,wish,protect|Sassy|4,,252,,252,|||||]\
        Ting-Lu||heavydutyboots|vesselofruin|earthquake,stealthrock,whirlwind,rest|Impish|252,4,252,,,|||||]\
        Toxapex||heavydutyboots|regenerator|toxic,banefulbunker,recover,haze|Calm|252,,4,,252,|||||]\
        Blissey||heavydutyboots|naturalcure|seismictoss,soft-boiled,calmmind,protect|Calm|4,,252,,252,|||||]\
        Dondozo||heavydutyboots|unaware|avalanche,rest,sleeptalk,curse|Impish|248,,252,,8,|||||"
    teams = ["https://pokepast.es/ad7a9a738ec1a82a", "https://pokepast.es/f09ff66281cee40e", "https://pokepast.es/4972596fc1ed58c2", "https://pokepast.es/1214a19de0fe9ba7"]
    with open("./asciikarp.txt") as ascii:
        ascii_lines = ascii.readlines()
        for i in ascii_lines:
            print(i, end='')
    print("Logging in.")
    useTeams = {
            "gen9ou" : GEN9_USE_TEAM,
            "gen3ou" : pokepasteToPacked(choice(teams))
        }
    sd = ShowdownConnection(USERNAME, PASSWORD, timeout=CONFIG["websocket"]["timeout"])
    # 
    # loginThread.start()
    # loginThread.join()

    sd.Start()
    while not sd._exit:
        pass