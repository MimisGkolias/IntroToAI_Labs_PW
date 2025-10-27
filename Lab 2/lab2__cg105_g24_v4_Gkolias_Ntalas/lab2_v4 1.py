import time
import copy

def validate_puzzle(puzzle):
    """
    Validates that the given Sudoku puzzle does not have any conflicts
    in its pre-filled cells (i.e., duplicates in any row, column, or 3x3 box).
    
    Returns True if valid, False if any conflicts are found.
    """
    # Check rows
    for i, row in enumerate(puzzle):
        nonzeros = [num for num in row if num != 0]
        if len(nonzeros) != len(set(nonzeros)):
            print(f"Conflict found in row {i+1}: {row}")
            return False

    # Check columns
    for c in range(9):
        col = [puzzle[r][c] for r in range(9) if puzzle[r][c] != 0]
        if len(col) != len(set(col)):
            print(f"Conflict found in column {c+1}: {col}")
            return False

    # Check 3x3 boxes
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            box = []
            for r in range(box_r, box_r+3):
                for c in range(box_c, box_c+3):
                    if puzzle[r][c] != 0:
                        box.append(puzzle[r][c])
            if len(box) != len(set(box)):
                print(f"Conflict found in 3x3 box starting at ({box_r+1},{box_c+1}): {box}")
                return False

    return True


class CSP:
    def __init__(self, variables, domains, constraints):
        """
        Initialize the CSP for solving Sudoku.
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None
        self.viz = []  # To store visualization steps

    def print_sudoku(self, puzzle):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - ")
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                print(puzzle[i][j], end=" ")
            print()
        print("\n")

    def forward_checking(self, var, value, assignment):
        """
        Remove the value from the domains of free variables that are in the constraints of var.
        Returns a tuple (success, pruned_values), where success is False if any domain becomes empty.
        """
        r, c = var
        pruned_values = []  # Store removed values for restoration

        for (x, y) in self.variables:
            if (x == r or y == c or (x // 3 == r // 3 and y // 3 == c // 3)) and (x, y) not in assignment:
                if value in self.domains[(x, y)]:
                    self.domains[(x, y)].remove(value)
                    pruned_values.append((x, y, value))
                    if not self.domains[(x, y)]:  # Domain is empty, so backtrack.
                        return False, pruned_values

        return True, pruned_values

    def is_valid(self, var, value, assignment):
        """
        Check if assigning 'value' to 'var' is valid according to Sudoku constraints.
        """
        r, c = var
        for (x, y) in assignment:
            if assignment[(x, y)] == value:
                if x == r or y == c or (x // 3 == r // 3 and y // 3 == c // 3):
                    return False
        return True

    def backtrack(self, assignment):
        """
        Backtracking algorithm to solve the puzzle.
        Returns a complete assignment dictionary if successful, or None if no solution exists.
        """
        if len(assignment) == len(self.variables):  # All variables assigned
            return assignment

        var = min(self.variables, key=lambda v: len(self.domains[v]) if v not in assignment else float('inf'))
        for value in self.least_constraining_value(var):
            if self.is_valid(var, value, assignment):
                assignment[var] = value
                self.viz.append((var, value))  # Store for visualization

                success, pruned_values = self.forward_checking(var, value, assignment)
                if success:
                    result = self.backtrack(assignment)
                    if result:
                        return result

                # Backtrack: restore pruned values and remove the assignment
                for x, y, v in pruned_values:
                    self.domains[(x, y)].add(v)
                self.domains[var] = {v for v in range(1, 10) if self.is_valid(var, v, assignment)}
                del assignment[var]

        return None

    def least_constraining_value(self, var):
        """
        Return the domain values of var, sorted by how few conflicts they cause.
        """
        return sorted(self.domains[var], key=lambda v: self.count_conflicts(var, v))

    def count_conflicts(self, var, value):
        """
        Count how many variables would lose this value from their domain if we assign value to var.
        """
        r, c = var
        conflict_count = 0
        for (x, y) in self.variables:
            if (x == r or y == c or (x // 3 == r // 3 and y // 3 == c // 3)) and (x, y) != var:
                if value in self.domains[(x, y)]:
                    conflict_count += 1
        return conflict_count

    def solve(self):
        """
        Solve the Sudoku puzzle using backtracking.
        """
        assignment = {}
        self.solution = self.backtrack(assignment)
        return self.solution, self.viz

    def visualize(self):
        """
        Visualize each step's board state with step numbers.
        """
        if not self.viz:
            print("No steps to visualize.")
            return

        board = [[puzzle[i][j] for j in range(9)] for i in range(9)]  # Copy original puzzle

        for step, (var, value) in enumerate(self.viz, 1):
            row, col = var
            board[row][col] = value  # Update visualization board
            print('#' * 8 + f' Step {step} ' + '#' * 8)
            self.print_sudoku(board)


# Example puzzle
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0], 
	[0, 0, 0, 1, 0, 5, 0, 0, 0], 
	[0, 9, 8, 0, 0, 0, 0, 6, 0], 
	[0, 0, 0, 0, 0, 3, 0, 0, 1], 
	[0, 0, 0, 0, 0, 0, 0, 0, 6], 
	[0, 0, 0, 0, 0, 0, 2, 8, 0], 
	[0, 0, 0, 0, 0, 0, 0, 0, 8], 
	[0, 0, 0, 0, 0, 0, 0, 1, 0], 
	[0, 0, 0, 0, 0, 0, 4, 0, 0]
]

# Validate the puzzle before solving.
if not validate_puzzle(puzzle):
    print("The initial puzzle has conflicting clues! Please fix them before solving.")
else:
    # Prepare variables, domains, and constraints for the CSP.
    variables = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    domains = {}
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                used_values = set(
                    puzzle[r] +  # Row values
                    [puzzle[i][c] for i in range(9)] +  # Column values
                    [puzzle[i][j] for i in range(r//3 * 3, (r//3 + 1) * 3)
                     for j in range(c//3 * 3, (c//3 + 1) * 3)]  # Box values
                )
                domains[(r, c)] = set(range(1, 10)) - used_values

    constraints = []
    for r in range(9):
        constraints.append([(r, c) for c in range(9)])
    for c in range(9):
        constraints.append([(r, c) for r in range(9)])
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            constraints.append([(box_r + r, box_c + c) for r in range(3) for c in range(3)])

    # Solve and display results.
    print('*' * 7, 'Solution', '*' * 7)
    csp = CSP(variables, domains, constraints)
    sol, viz = csp.solve()

    if sol:
        solved_puzzle = [
            [puzzle[i][j] if (i, j) not in sol else sol[(i, j)] for j in range(9)]
            for i in range(9)
        ]
        csp.print_sudoku(solved_puzzle)
        csp.visualize()
    else:
        print("Solution does not exist")
