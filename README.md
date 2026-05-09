# Dual Simplex Method Solver (Двоїстий симплекс-метод)

## About the Project

This is a standalone web application designed to solve Linear Programming (LP) problems using the Dual Simplex Method (Двоїстий симплекс-метод). The primary goal of this project is to assist university students in understanding and completing their operations research laboratory works. Built entirely with static frontend technologies, the application requires no backend backend and is ready to be hosted directly on GitHub Pages.

## Key Features

* **Step-by-Step Educational Solutions:** The application goes beyond just providing the final answer; it breaks down the entire algorithm step-by-step to facilitate learning.


* **Detailed Simplex Tables:** Automatically generates and renders intermediate simplex tables for every iteration of the mathematical process.


* **Visual Highlighting:** Clearly highlights the pivot row, pivot column, and pivot element (розв'язувальний елемент) in each table, making it easy for students to track the algorithm's logic.


* **Clear Explanations:** Provides brief, descriptive text explanations of the actions taken between each table.


* **Dynamic Input:** Allows users to easily input objective functions and dynamically add or remove constraints to match their specific laboratory tasks.

## Technologies Used

* **HTML5:** For structuring the application, dynamic input forms, and data tables.

* **CSS3:** For a clean, academic, and readable user interface.

* **Vanilla JavaScript (ES6+):** Contains the core mathematical logic for the Dual Simplex algorithm and handles DOM manipulation without the need for heavy external libraries.

## How It Works

1. **Input:** The user inputs the objective function coefficients and selects whether to maximize or minimize. They then input the constraint matrix, including the inequality/equality signs (≤, ≥, =) and the right-hand side values.
2. **Preparation:** The JavaScript engine converts the problem into an "Almost Canonical Form" (Майже канонічна форма), introducing slack variables and adjusting signs as required by the Dual Simplex rules.
3. **Iteration:** The application calculates the initial evaluations and iteratively applies the Jordan-Gauss elimination method.
4. **Result:** The UI renders each step until an optimal solution is reached (or determines if the problem has no valid solution), ultimately displaying the optimal variables vector and the extremum value of the objective function.
