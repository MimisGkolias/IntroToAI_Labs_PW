


class CSP: 
	def __init__(self,variables, domains, constraints): 
		"""
		Initialization of the CSP class

		Parameters:
		- variables
		- domains
		- constraints

		Objective:
		- solution: sudoku solution
		- viz: everything needed for visualization
		"""
		self.variables = variables 
		self.domains = domains 
		self.constraints = constraints 
		self.solution = None
		self.viz = None


	def print_sudoku(puzzle): 
		for i in range(9): 
			if i % 3 == 0 and i != 0: 
				print("- - - - - - - - - - - ") 
			for j in range(9): 
				if j % 3 == 0 and j != 0: 
					print(" | ", end="") 
				print(puzzle[i][j], end=" ") 
			print() 

	def get_constraints():
		constraints = []
		for r in range(9):

	def visualize(self)

	def solve(self): 
		assignment = {} 
		self.solution = self.backtrack(assignment) 
		return self.solution 
	
	def forward_checking(self, var, value, assignment):
		"""
		Function that removes the value from the domains of free variables that are in the constraints of the var

		Parameters:
		- var: variable that was assigned the value
		- value: value that was assigned to the variable
		- assignment: dict with all the assignments to the variables

		"""
	def backtrack(self, assignment): 
		"""
		Backtracking algorithm

		Parameters:
		- assignment: dict with all the assignments to the variables

		Returns:
		- assignment: dict with all the assigments to the variables, or None if solution is not found. Return the first found solution
		"""



puzzle = [[5, 3, 0, 0, 7, 0, 0, 0, 0], 
		  [0, 0, 0, 1, 0, 5, 0, 0, 0], 
		  [0, 9, 8, 0, 0, 0, 0, 6, 0], 
		  [0, 0, 0, 0, 0, 3, 0, 0, 1], 
		  [0, 0, 0, 0, 0, 0, 0, 0, 6], 
		  [0, 0, 0, 0, 0, 0, 2, 8, 0], 
		  [0, 0, 0, 0, 0, 0, 0, 0, 8], 
		  [0, 0, 0, 0, 0, 0, 0, 1, 0], 
		  [0, 0, 0, 0, 0, 0, 4, 0, 0] 
		] 	
# Based on the puzzle create variables, domains, and constraints for initialization of CSP class


print('*'*7,'Solution','*'*7) 
csp = CSP(variables, domains, constraints) 
sol, viz = csp.solve() 
csp.print_sudoku(puzzle)
solution = [[0 for i in range(9)] for i in range(9)] 
if sol is not None:
	for i,j in sol: 
		solution[i][j]=sol[i,j] 
		
	csp.print_sudoku(solution)
	csp.visualize()
else:
	print("Solution does not exist")
