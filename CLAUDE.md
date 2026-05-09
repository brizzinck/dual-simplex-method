# CLAUDE.md

# Interactive Trainer: Dual Simplex Method

## 📌 About the Project
This web application is an interactive educational trainer designed to help university students master the **Dual Simplex Method** for solving Linear Programming problems. 

Unlike standard calculators that instantly output the final answer, this application is specifically built for completing and verifying laboratory works in Operations Research. The system calculates the correct values in the background but forces the student to make decisions and perform the mathematical calculations manually at each step of the algorithm.

## ✨ Key Features
- **Interactive Step-by-Step Learning:** The student navigates through all stages of solving the problem manually, controlling the entire process.
- **Optimality Check:** Users must analyze the table and determine whether the current plan is optimal (checking if all $b_i \ge 0$).
- **Selecting Pivot Elements:** Interactive, clickable tables allow students to choose the pivot row (most negative $b_i$) and the pivot column (minimum ratio $|\Delta_j / a_{rj}|$). The app will display an error if the choice violates the method's mathematical rules.
- **Manual Recalculation (Gauss-Jordan Elimination):** At each iteration, the app generates a new table with empty input fields. The student must calculate the new values on scratch paper and type them into the program.
- **Instant Validation:** User inputs are automatically validated against the correct background calculations (accounting for a small floating-point `epsilon` error margin). Correct cells are highlighted in green, and incorrect ones in red.
- **Hint System:** If a student gets stuck in the calculations, they can use the "Help/Reveal" button to show the correct values and continue solving the laboratory work.

## 🛠 Technologies Used
The project is built entirely with core web technologies, ensuring a lightweight footprint without heavy frameworks or external dependencies.
- **HTML5:** For the dynamic generation of matrices, tables, and input forms.
- **CSS3:** For a clean, modern, and academic UI design. Features intuitive color-coded highlights (success/error states, active rows/columns, pivot elements).
- **Vanilla JavaScript (ES6+):** Contains the math engine (`MathCore`) for solving the problem in the background, a step-by-step state manager (`UI/State Manager`), and user input validation logic.

## 🚀 How to Run Locally
Since the project does not require a backend and consists solely of static files, running it is extremely simple:
1. Clone the repository or download the source files (`index.html`, `style.css`, `script.js`).
2. Place all files in a single folder.
3. Double-click the `index.html` file to open it in any modern web browser (Chrome, Firefox, Safari, etc.).
4. The application is ready to use!

## 🌐 Deployment
This project is fully optimized for free hosting on **GitHub Pages**.
1. Upload the files to a new GitHub repository.
2. Go to the repository `Settings` -> `Pages` menu.
3. Under the `Build and deployment` section (Source: Deploy from a branch), select the `main` (or `master`) branch and click `Save`.
4. Wait 1-2 minutes, and you will receive a public link to your live interactive trainer.
