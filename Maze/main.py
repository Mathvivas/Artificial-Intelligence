import sys
class Node():
    # Construtor de Node
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

# Pilha - last in first out
# Depth First Search - DFS
# Esgota um caminho para depois pegar outro
class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier!")
        else:
            node = self.frontier[-1]    # Último item da lista
            # Nova lista --> Todos os valores anteriores, exceto o último
            self.frontier = self.frontier[:-1] # Não pega o último
            return node

# Lista - First in First out
# Breadth First Search - BFS
# Observa o mesmo nível inteiro, antes de passar para o próximo:
# Todos do primeiro andar, depois todos do segundo, depois todos do terceiro...
class QueueFrontier(StackFrontier):
    # QueueFrontier herda (extends - Java) de StackFrontier

    # @Override
    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier!")
        else:
            node = self.frontier[0]
            # Nova lista --> Todos os valores a partir do segundo
            self.frontier = self.frontier[1:] # Não pega o primeiro
            return node

class Maze():
    def __init__(self, filename):
        with open(filename) as f:
            contents = f.read()

        # Validação do início e fim
        if contents.count("A") != 1:
            raise Exception("Maze must have exactly one start point!")
        if contents.count("B") != 1:
            raise Exception("Maze must have exactly one goal!")

        # Determina altura e largura do labirinto
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Paredes
        self.walls = []
        for i in range(self.height):    # Linhas
            row = []
            for j in range(self.width): # Colunas
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve(self):
        """Finds a solution to maze, if one exists."""

        # Número de estados explorados
        self.num_explored = 0

        # Inicializa a fronteira com o ponto inicial
        start = Node(state=self.start, parent=None, action=None)
        frontier = QueueFrontier()
        #frontier = StackFrontier()
        frontier.add(start)

        # Inicializa um explored set vazio
        self.explored = set()

        # Looping até achar a solução
        while True:

            # Se não sobrar nada na fronteira, sem caminho
            if frontier.empty():
                raise Exception("no solution")

            # Escolha um node na fronteira
            node = frontier.remove()
            self.num_explored += 1

            # Se o node é o goal, então é a solução
            if node.state == self.goal:
                actions = []
                cells = []

                # Segue o parent para achar a solução
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # Marca o node como explorado
            self.explored.add(node.state)

            # Adiciona os vizinhos na fronteira
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                        ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)
        img.show()



m = Maze("maze2.txt")
print("Maze:")
m.print()
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image("maze.png", show_explored=True)
# Caminho Vermelho --> Explorado
# Caminho Branco --> Não Explorado