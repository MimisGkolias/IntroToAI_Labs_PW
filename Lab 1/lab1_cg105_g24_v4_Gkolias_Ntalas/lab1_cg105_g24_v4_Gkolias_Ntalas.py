import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp

x_sym, y_sym = sp.symbols('x y')

def function(x, y):
    return 2*np.sin(x)+3*np.cos(y)

def symbolic_function(x, y):
    return 2*sp.sin(x) + 3*sp.cos(y)

def gradient(f, variables):
    grad_expr = [sp.diff(f, var) for var in variables]
    grad_func = sp.lambdify(variables, grad_expr, 'numpy')
    return grad_func

def hessian(f, variables):
    grad_expr = [sp.diff(f, var) for var in variables]
    hessian_expr = [[sp.diff(g, var) for var in variables] for g in grad_expr]
    hessian_func = sp.lambdify(variables, hessian_expr, 'numpy')
    return hessian_func

def newton_method(initial_guess, alpha, tol=1e-6, max_iter=1000):
    """
    Newton method
    
    Parameters:
    - initial_guess: initial 2D coordinate vector
    - alpha: step size parameter
    - tol: tolerance, convergence criteria
    - max_iter: maximum number of iterations

    """
    grad_func = gradient(symbolic_function(x_sym, y_sym), [x_sym, y_sym])
    hess_func = hessian(symbolic_function(x_sym, y_sym), [x_sym, y_sym])
    x_k = np.array(initial_guess, dtype=float)
    iter_count = 0
    path = [x_k.copy()]
    
    for _ in range(max_iter):
        grad = np.array(grad_func(x_k[0], x_k[1]))
        hess = np.array(hess_func(x_k[0], x_k[1]))
        
        if np.linalg.det(hess) == 0:
            print("Hessian is singular, stopping optimization.")
            break
        
        step = np.linalg.solve(hess, -grad)
        x_k += alpha * step
        path.append(x_k.copy())
        iter_count += 1
        
        if np.linalg.norm(step) < tol:
            break
    
    return x_k, iter_count, np.array(path)





def visualize(path, initial_guess, step_size):
    """
    Visualization function: creates 3D plot of the function. Use colors to show the Z-coordinate.
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)

    # Convert path to numpy array
    path = np.array(path)

    # Scatter plot for optimization path
    ax.scatter(path[:, 0], path[:, 1], function(path[:, 0], path[:, 1]),
               color='r', marker='o', s=50, label="Optimization Path")
    
    # Highlight the initial and final points with different colors
    ax.scatter(initial_guess[0], initial_guess[1], function(initial_guess[0], initial_guess[1]),
               color='g', marker='o', s=100, label="Initial Point")
    
    ax.scatter(path[-1, 0], path[-1, 1], function(path[-1, 0], path[-1, 1]),
               color='b', marker='o', s=100, label="Ending Point")
    
    # Annotate the final points with text    
    ax.text(path[-1, 0], path[-1, 1], function(path[-1, 0], path[-1, 1]),
            f'({path[-1, 0]:.2f}, {path[-1, 1]:.2f})', color='b', fontsize=12)
    

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value (Z)')

    # Dynamically change the title to include initial guess and step size
    plt.title(f"Optimization Path (Init: {initial_guess}, Step: {step_size})")
    ax.legend()

    plt.show()


#Example usage:
initial_guesses = [[-3, -3], [-1,-1], [1,1], [3,3], [-3, -4], [0, 0]
step_sizes = [0.01, 0.1, 1.0, 2.0]

for initial_guess in initial_guesses:
    for step_size in step_sizes:
        minimum, iterations, path = newton_method(initial_guess, step_size)

        print(f"Minimum approximation with initial guess {initial_guess} and learning rate {step_size}: {minimum}, Iterations: {iterations}")

        # visualize(path, initial_guess, step_size)
