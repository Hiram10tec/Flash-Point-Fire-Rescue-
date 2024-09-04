from flask import Flask, jsonify
from mesa import Agent, Model
import json

# Debido a que necesitamos que existe un solo agente por celda, elegimos
# ''SingleGrid''.
from mesa.space import MultiGrid

# Con ''SimultaneousActivation, hacemos que todos los agentes se activen
# ''al mismo tiempo''.
from mesa.time import RandomActivation

# Haremos uso de ''DataCollector'' para obtener información de cada paso
# de la simulación.
from mesa.datacollection import DataCollector

import random

# matplotlib lo usaremos crear una animación de cada uno de los pasos
# del modelo.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

# Importamos los siguientes paquetes para el mejor manejo de valores
# numéricos.
import numpy as np
import pandas as pd

# Definimos otros paquetes que vamos a usar para medir el tiempo de
# ejecución de nuestro algoritmo.
import time
import datetime

def get_grid(model):
    grid = np.copy(model.matriz_pared)
    for i in range(model.grid.width):
        for j in range(model.grid.height):
            if model.matriz_fuego[i][j] == 1:
                grid[i][j] = 9
            if model.matriz_fuego[i][j] == 2:
                grid[i][j] = 10
    for agent in model.schedule.agents:
        if agent.pos is not None:
            grid[agent.pos[0]][agent.pos[1]] = 11
    return grid

class Nodo():
    def __init__(self, data):
        self.data = data
        self.next = None

class Fila():
    def __init__(self):
        self.head = None
        self.tail = None
        self.pointer = None
        self.size = 0

    def push(self, data):
        nodo = Nodo(data)
        if self.size == 0:
            self.head = nodo
            self.pointer = self.head
        else:
            self.tail.next = nodo
        nodo.next = self.head
        self.tail = nodo
        self.size += 1

    def pop(self, data):
        if self.size == 0:
            return
        p = self.head
        q = self.head.next
        if p.data == data:
            self.head = q
            self.tail.next = self.head
        else:
            while q.data != data:
                p = p.next
                q = q.next
            p.next = q.next
            if q == self.tail:
                self.tail = p
        self.size -= 1

class Building(Model):
    def __init__(self, largo, ancho, num_agents, paredes, puertas, entradas):
        super().__init__()
        self.grid = MultiGrid(largo*2 + 3, ancho*2 + 3, torus = False)
        self.schedule = RandomActivation(self)

        self.datacollector = DataCollector(
            model_reporters = {
                "Grid": get_grid
            }
        )

        self.running = True
        self.matriz_pared = np.zeros((largo*2 + 3, ancho*2 + 3))
        self.entradas = np.array(entradas)*2
        self.matriz_fuego = np.zeros((largo*2 + 3, ancho*2 + 3))
        self.agentes_vivos = Fila()

        for i in range(1, largo*2 + 2, 2):
            for j in range(ancho*2 + 3):
                self.matriz_pared[i][j] = 1

        for i in range(largo*2 + 3):
            for j in range(1, ancho*2 + 2, 2):
                self.matriz_pared[i][j] = 1

        for i in range(largo):
            for j in range(ancho):
                if paredes[i][j][0] == 1:
                    self.matriz_pared[i*2 + 1][j*2 + 2] = 3
                if paredes[i][j][1] == 1:
                    self.matriz_pared[i*2 + 2][j*2 + 1] = 3
                if paredes[i][j][2] == 1:
                    self.matriz_pared[i*2 + 3][j*2 + 2] = 3
                if paredes[i][j][3] == 1:
                    self.matriz_pared[i*2 + 2][j*2 + 3] = 3

                if paredes[i][j][0] == 1 and paredes[i][j][1] == 1:
                    self.matriz_pared[i*2 + 1][j*2 + 1] = 3
                if paredes[i][j][0] == 1 and paredes[i][j][3] == 1:
                    self.matriz_pared[i*2 + 1][j*2 + 3] = 3
                if paredes[i][j][2] == 1 and paredes[i][j][1] == 1:
                    self.matriz_pared[i*2 + 3][j*2 + 1] = 3
                if paredes[i][j][2] == 1 and paredes[i][j][3] == 1:
                    self.matriz_pared[i*2 + 3][j*2 + 3] = 3

        for i in range(1, largo*2 + 2, 2):
            for j in range(1, ancho*2 + 2, 2):
                if i > 0 and i < largo*2:
                    if self.matriz_pared[i-1][j] == 3 and self.matriz_pared[i+1][j] == 3:
                        self.matriz_pared[i][j] = 3
                if j > 0 and j < ancho*2:
                    if self.matriz_pared[i][j-1] == 3 and self.matriz_pared[i][j+1] == 3:
                        self.matriz_pared[i][j] = 3

        for i in range(len(puertas)):
            if puertas[i][0] == puertas[i][2]:
                self.matriz_pared[puertas[i][0]*2][puertas[i][1]*2 + 1] = 4
            if puertas[i][1] == puertas[i][3]:
                self.matriz_pared[puertas[i][0]*2 + 1][puertas[i][1]*2] = 4

        for i in range(len(self.entradas)):
            if self.entradas[i][0] == 2:
                self.matriz_pared[1][self.entradas[i][1]] = 6
            if self.entradas[i][0] == self.grid.width - 3:
                self.matriz_pared[self.grid.width - 2][self.entradas[i][1]] = 6
            if self.entradas[i][1] == 2:
                self.matriz_pared[self.entradas[i][0]][1] = 6
            if self.entradas[i][1] == self.grid.height - 3:
                self.matriz_pared[self.entradas[i][0]][self.grid.height - 2] = 6


        for i in range(len(fuegos)):
            self.matriz_fuego[fuegos[i][0]*2][fuegos[i][1]*2] = 2


        for i in range(num_agents):
            a = Bombero(i, self)
            x, y = self.random.choice(self.entradas)
            self.grid.place_agent(a, (x,y))
            self.schedule.add(a)
            self.agentes_vivos.push(a)

    def step(self):
        self.datacollector.collect(self)
        if self.agentes_vivos.size == 0:
            self.running = False
            return
        agente = self.agentes_vivos.pointer.data
        agente.step()
        if agente.end_turn:
            agente.action_points = min(agente.action_points + 4, 8)
            self.agentes_vivos.pointer = self.agentes_vivos.pointer.next

class Bombero(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.carrying_victim = False
        self.action_points = 4
        self.last_pos = None
        self.afuera = False
        self.end_turn = False

    def step(self):
        if self.pos is None:
            return

        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        door_steps = [step for step in possible_steps
                      if self.model.matriz_pared[step[0]][step[1]] == 4
                      or self.model.matriz_pared[step[0]][step[1]] == 5]
        possible_door = [step for step in door_steps
                         if self.action_points > 0]
        wall_steps = [step for step in possible_steps
                      if self.model.matriz_pared[step[0]][step[1]] == 3
                      or self.model.matriz_pared[step[0]][step[1]] == 2]
        possible_wall = [step for step in wall_steps
                         if self.action_points > 1]

        if possible_door and possible_wall:
            choice = self.random.choice([i for i in range(9)])
            if choice < 6 and self.action_points > 1:
                self.random_move()
            elif 5 < choice < 8:
                self.flip_door(self.random.choice(door_steps))
            else:
                self.chop_wall(self.random.choice(wall_steps))

        elif possible_door:
            choice = self.random.choice([0,1,2])
            if choice < 2 and self.action_points > 1:
                self.random_move()
            else:
                self.flip_door(self.random.choice(door_steps))

        elif possible_wall:
            choice = self.random.choice([i for i in range(7)])
            if choice < 6 and self.action_points > 1:
                self.random_move()
            else:
                self.chop_wall(self.random.choice(wall_steps))

        elif self.action_points > 1:
            self.random_move()

        else:
            self.end_turn = True


    def random_move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        valid_steps = [step for step in possible_steps
                       if self.model.matriz_pared[step[0]][step[1]] != 2
                       and self.model.matriz_pared[step[0]][step[1]] != 3
                       and self.model.matriz_pared[step[0]][step[1]] != 4]


        if valid_steps:
            new_pos = self.random.choice(valid_steps)

            if self.afuera:
                self.model.grid.move_agent(self, self.last_pos)
                self.afuera = False

            elif new_pos == (self.pos[0] - 1, self.pos[1]):
                new_pos = (self.pos[0] - 2, self.pos[1])
                if new_pos[0] == 0:
                    self.afuera = True
                    self.last_pos = self.pos
                self.model.grid.move_agent(self, new_pos)


            elif new_pos == (self.pos[0] + 1, self.pos[1]):
                new_pos = (self.pos[0] + 2, self.pos[1])
                if new_pos[0] == self.model.grid.width - 1:
                    self.afuera = True
                    self.last_pos = self.pos
                self.model.grid.move_agent(self, new_pos)


            elif new_pos == (self.pos[0], self.pos[1] - 1):
                new_pos = (self.pos[0], self.pos[1] - 2)
                if new_pos[1] == 0:
                    self.afuera = True
                    self.last_pos = self.pos
                self.model.grid.move_agent(self, new_pos)


            elif new_pos == (self.pos[0], self.pos[1] + 1):
                new_pos = (self.pos[0], self.pos[1] + 2)
                if new_pos[1] == self.model.grid.height - 1:
                    self.afuera = True
                    self.last_pos = self.pos
                self.model.grid.move_agent(self, new_pos)

            if self.model.matriz_fuego[new_pos[0]][new_pos[1]] == 2:
                self.action_points -= 2
            else:
                self.action_points -= 1


    def flip_door(self, pos):
        if self.model.matriz_pared[pos[0]][pos[1]] == 5:
            self.model.matriz_pared[pos[0]][pos[1]] = 4
        elif self.model.matriz_pared[pos[0]][pos[1]] == 4:
            self.model.matriz_pared[pos[0]][pos[1]] = 5

        self.action_points -= 1


    def chop_wall(self, pos):
        if self.model.matriz_pared[pos[0]][pos[1]] == 3:
            self.model.matriz_pared[pos[0]][pos[1]] = 2
        elif self.model.matriz_pared[pos[0]][pos[1]] == 2:
            self.model.matriz_pared[pos[0]][pos[1]] = 1

        self.action_points -= 2

paredes = [
    [[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]],
    [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
    [[0, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]],
    [[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]],
    [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 1, 1]]
]

puertas = [
    [1, 3, 1, 4],
    [2, 5, 2, 6],
    [2, 8, 3, 8],
    [3, 2, 3, 3],
    [4, 4, 5, 4],
    [4, 6, 4, 7],
    [6, 5, 6, 6],
    [6, 7, 6, 8]
]

entradas = [
    [1, 6],
    [3, 1],
    [4, 8],
    [6, 3]
]

fuegos = [
    [2, 2],
    [2, 3],
    [3, 2],
    [3, 3],
    [3, 4],
    [3, 5],
    [4, 4],
    [5, 6],
    [5, 7],
    [6, 6]
]


num_agents = 6
largo = 6
ancho = 8

start_time = time.time()

model = Building(largo, ancho, num_agents, paredes, puertas, entradas)

for i in range(200):
    model.step()

print('Tiempo de ejecución:',
      str(datetime.timedelta(seconds=(time.time() - start_time))))

all_grid = model.datacollector.get_model_vars_dataframe()

data_df = pd.DataFrame(all_grid)
data_json = data_df.to_json()

app = Flask(__name__)
@app.route("/", methods={"GET"})
def index():
    return jsonify(data_json)


if __name__ == "__main__":
    app.run(debug=True)