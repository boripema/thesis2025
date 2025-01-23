import mesa
import random
from enum import Enum 

from mesa.time import RandomActivation


class Symbol(Enum):
    HUNTER_A = 1
    HUNTER_B = 2
    STAG = 3

    def symbol_to_entity(symbol):
        if symbol == str(Symbol.HUNTER_A):
            return HunterA
        elif symbol == str(Symbol.HUNTER_B):
            return HunterB
        elif symbol == str(Symbol.STAG):
            return Stag
        else: 
            raise RuntimeError(f"Unknown symbol {symbol}.")

    def __str__(self):
        if self == Symbol.HUNTER_A:
            return "☻"
        if self == Symbol.HUNTER_B:
            return "☺"
        if self == Symbol.STAG:
            return "♠"
        else:
            raise RuntimeError(f"Unknown symbol {self}.")
        
    def __repr__(self):
        return str(self)
    
        
class Stag(mesa.Agent):

    def __init__(self, unique_id, model, position, meat=15):
        super().__init__(unique_id, model)
        self.meat = meat

    def step(self): 
        pass

    def show(self):
        return str(Symbol.STAG)
    
    def destroy(self):
        regen_time = self.model.schedule.time + 10
        self.model.regeneration_queue.append((self.__class__, regen_time, self.pos))
        self.model.remove_entity(self)


class AgentBody(mesa.Agent):

    def __init__(self, unique_id, model, position, satiety):
        super().__init__(unique_id, model)
        self.mental_init()
        self.is_alive = True

    def mental_init(self):
        self.next_action = None

    @staticmethod
    def get_directions():
        directions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                directions.append((i, j))
        return directions
    
    
    def get_percepts(self, radius=2):

        relevant_entities = self._get_relevant_entities()

        if relevant_entities is None:
            raise RuntimeError("Define relevant entities.")

        percepts_about = {}
        for entity_type in relevant_entities:
            percepts_about[entity_type] = [0] * (2 * radius + 1) ** 2

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                absx, absy = self.pos
                x = (absx + dx) % self.model.width
                y = (absy + dy) % self.model.height

                grid_index = (dx + radius) * (2 * radius + 1) + (dy + radius)

                for item in self.model.grid.get_cell_list_contents((x, y)):
                    for entity_type in relevant_entities:
                        if isinstance(item, entity_type):  # Correct comparison
                            percepts_about[entity_type][grid_index] += 1
    
        flat_percepts = []
        sorted_keys = sorted(percepts_about.keys(), key=lambda k: k.__name__)

        for key in sorted_keys:
            flat_percepts += percepts_about[key] 

        flat_percepts.append(self.satiety / self.MAX_SATIETY)
        
        for agent in self.model.entities:
            if isinstance(agent, AgentBody):
                if agent != self:
                    flat_percepts.append(agent.satiety / self.MAX_SATIETY)

        flat_percepts.append(self.model.nstags / self.model.total_stags)

        return flat_percepts
    

    def move(self, direction):
        dx, dy = direction
        absx, absy = self.pos
        x = (absx + dx) % self.model.width
        y = (absy + dy) % self.model.height
        if mesa.Agent in self.model.disabilities:
            disabilities = self.model.disabilities[mesa.Agent]["move"]
        else:
            disabilities = []
        if (x, y) not in (disabled() for disabled in disabilities):
            self.model.grid.move_agent(self, (x, y))
        else:
            self.model.event.append((self, False))
            self.trace("action 'move' failed")

    def step(self):
        if not self.is_alive:
            return
        self.satiety -= 1
        self.mental_step()
        self.react()

    def mental_step(self):
        percepts = self.get_percepts
        if self.next_action is not None:
            self.trace(f"next action: {self.next_action}")
            self.move(self.next_action)
            self.next_action = None

    def trace(self, text):
        state = self.get_percepts()
        self.model.console.append(f"{type(self).__name__} {self.unique_id} > {text}")
        
    def destroy(self):
        pass

    def _get_relevant_entities(self):
        return None

    def show(self):
        pass
    
    def react(self):
        elems = self.model.grid.get_cell_list_contents([self.pos])
        for elem in elems:
            if elem != self:
                if type(elem) == Stag:
                    self.model.events.append((self, "hunted"))
                    self.satiety += elem.meat
                    self.trace("I've found a stag!")
                    elem.destroy()
                    if self.model.nstags == 0:
                        self.model.events.append((self, "extinction"))
        self.satiety = min(self.satiety, self.MAX_SATIETY)

    def starve(self):
        self.is_alive = False
        
    
class HunterA(AgentBody):
    MAX_SATIETY = 30
    
    def __init__(self, unique_id, model, position, satiety=MAX_SATIETY):
        super().__init__(unique_id, model, position, satiety)
        self.satiety = satiety
        self.is_alive = True

    def starve(self):
        self.is_alive = False


    def _get_relevant_entities(self):
        return [Stag, HunterB]
    
    def show(self):
        return str(Symbol.HUNTER_A)
    
    def step(self):
        if not self.is_alive:
            return
        super().step()
    

class HunterB(AgentBody):
    MAX_SATIETY = 30

    def __init__(self, unique_id, model, position, satiety=MAX_SATIETY):
        super().__init__(unique_id, model, position, satiety)
        self.satiety = satiety
        self.is_alive = True

    def starve(self):
        self.is_alive = False

    def _get_relevant_entities(self):
        return [Stag, HunterA]

    def show(self):
        return str(Symbol.HUNTER_B)
    
    def step(self):
        if not self.is_alive:
            return
        super().step()

class WorldModel(mesa.Model):
    
    def __init__(self, width, height):
        self.entities = []
        self.disabilities = {}
        self.width = width
        self.height = height
        
        self.schedule = RandomActivation(self)

        self.grid = mesa.space.MultiGrid(width, height, True)
        self.end = False
        self.nstags = 0
        self.total_stags = 0
        self.regeneration_queue = []
        self.console = []

    def step(self):     # Calls the step from env.py
        self.events = []
        self.schedule.step()

        for entity in self.entities:
            if isinstance(entity, AgentBody) and entity.is_alive:
                if entity.satiety <= 0:  # End simulation if hunter's hunger is 0
                    self.events.append((entity, "starved"))
                    entity.starve()
    

        self.regenerate_entities()
        return self.end, self.events
    
    def get_positions(self):
        positions = {}
        size = self.width * self.height
        for entity in self.entities:
            entity_type = str(type(entity))
            if entity_type not in positions:
                positions[entity_type] = []
                for _ in range(size):
                    positions[entity_type].append(0)
            
            if entity.pos is not None:
                x, y = entity.pos
                positions[entity_type][y * self.width + x] += 1

        flat_positions = []
        for entity_type in sorted([str(k) for k in positions.keys()]):
            flat_positions += positions[entity_type]

        return flat_positions
    
    def add_entity(self, entity_type, x, y):
        if entity_type == HunterA:
            unique_id = 0
        elif entity_type == HunterB:
            unique_id = 1
        else:
            unique_id = len(self.entities) + 2
        entity = entity_type(unique_id, self, (x, y))
        self.grid.place_agent(entity, (x, y))
        self.schedule.add(entity)
        self.entities.append(entity)
        
        if isinstance(entity, Stag):
            self.nstags += 1
            self.total_stags = max(self.total_stags, self.nstags)

    def remove_entity(self, entity):
        self.grid.remove_agent(entity)
        self.schedule.remove(entity)

        if isinstance(entity, Stag):
            self.nstags -= 1
            if self.nstags == 0:
                self.end = True

    def regenerate_entities(self):  # ADDED
        regenerated_objects = []

        for obj, regen_time, pos in self.regeneration_queue:
            if self.schedule.time >= regen_time:  # Check if it's time to regenerate
                self.add_entity(obj, *pos)  # Add back to grid
                regenerated_objects.append((obj, regen_time, pos))  # Track regenerated objects

        self.regeneration_queue = [
            item for item in self.regeneration_queue if item not in regenerated_objects
        ]

    def remove_disability(self, entity_type, action, callable_for_value):
        self.disabilities[entity_type][action].remove(callable_for_value)

    
#######################
# helpers
#######################

def load_world(map):
    # remove trailing new line at the beginning
    if map[0] == "\n": map = map[1:]

    width = map.index("\n") - 2  # accounting for the borders
    if width == 0:
        raise ValueError("Unexpected dimensions of the map.")

    height = int(len(map) / (width + 3)) - 2  # accounting for the borders and newlines
    if height == 0 or len(map) % (width + 3) != 0:
        raise ValueError("Unexpected dimensions of the map.")

    # print(f"loading map on a {width}x{height} grid...")

    model = WorldModel(height, width)

    x = 0
    y = 0
    for z, ch in enumerate(map):
        if width + 3 < z < len(map) - width - 3:
            if 0 < z % (width + 3) <= width:
                if ch == " ":
                    pass
                else:
                    entity_type = Symbol.symbol_to_entity(ch)
                    model.add_entity(entity_type, x, y)
                y += 1
            if z % (width + 3) == 0:
                x += 1
                y = 0

    return model


def create_random_world(height, width, entities_dict):

    model = WorldModel(width, height)

    entity_types = entities_dict.keys()
    entities = []
    for type in entity_types:
        for i in range(entities_dict[type]):
            entities.append(type)

    map_size = width * height
    if len(entities) > map_size:
        raise RuntimeError("Not valid value: the map size should be bigger than the number of entities to be created.")

    x = 0; y = 0
    while y < height and x < width:
        generated_map_size = x + y*width
        if len(entities) == 0:
            break
        prob = len(entities)/(map_size - generated_map_size)
        if random.random() <= prob:
            selected = random.randint(0, len(entities) - 1)
            model.add_entity(entities[selected], x, y)
            del(entities[selected])
        x += 1
        if x >= width:
            x = 0
            y += 1

    return model


#######################
# viewer
#######################

import time


def move(x, y):
    print("\033[%d;%dH" % (y, x))


class WorldView:
    def __init__(self, world_model, fps=25):
        self.world = world_model

        # from frame per second (fps) toseconds per frame (ms)
        # eg. 25 f/s = 1/25 s/f = 1000/25 ms/f = 40 ms/f
        self.delay = 1/fps

    def init(self):
        print('\x1b[2J')

    def header(self):
        move(0, 0)
        print("mesagym -- minimal gym built on top on mesa\n")

    def show(self, reverse_order=False):
        self.header()
        string = ""

        string += "|"
        for i in range(0, self.world.height):
            string += "-"
        string += "|\n"

        # for how mesa furnish the coordinate
        # we have to print the transpose of the world
        for cell in self.world.grid.coord_iter():
            cell_content, (x, y) = cell

            if y == 0:
                string += "|"
            if len(cell_content) > 0:
                found = False
                for entity in cell_content:
                    if isinstance(entity, AgentBody):
                        string += entity.show()
                        found = True
                        break
                if not found:
                    string += cell_content[0].show()
            else:
                string += " "
            if y == self.world.height - 1:
                string += "|\n"

        string += "|"
        for i in range(0, self.world.height):
            string += "-"
        string += "|\n"

        print(string)

        print(">>> console <<<")

        if reverse_order:
            console = reversed(self.world.console[-5:])
        else:
            console = self.world.console[-5:]
        for item in console:
            print(item)

        time.sleep(self.delay)


