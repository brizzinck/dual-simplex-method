'use strict';

/* ═══════════════════════════════════════════════
   MathCore — Dual Simplex Method engine
   All tableau indices: row 0..m-1 are constraints,
   row m is the objective (Δ / reduced costs).
   Columns 0..n-1 are decision variables,
   columns n..n+m-1 are slack variables,
   column n+m is the RHS (b).
═══════════════════════════════════════════════ */
class DualSimplexSolver {
  static EPSILON = 1e-9;

  /* Eliminate floating-point noise: values within EPSILON of zero become 0 */
  static clean(v) { return Math.abs(v) < DualSimplexSolver.EPSILON ? 0 : v; }

  /* ── Construction ───────────────────────────
     c      : Float64Array(n)   objective coefficients
     A      : Float64Array(m*n) constraint matrix, row-major
     b      : Float64Array(m)   RHS
     signs  : string[]          '<=' | '>=' | '=' per row
     dir    : 'min' | 'max'

     Equality constraints ('=') are handled by identifying a decision variable
     whose column is a unit vector restricted to that row (coefficient ≠ 0 in
     the '=' row, zero in every other row).  That variable becomes the initial
     basic variable for the row, the row is normalised, and the objective row is
     updated via Gauss-Jordan so the reduced cost of the basic variable is zero.
     No slack column is added for '=' rows; mSlack ≤ m.
  ─────────────────────────────────────────── */
  constructor(c, A, b, signs, dir) {
    this.n   = c.length;
    this.m   = b.length;
    this.dir = dir;

    // Convert max → min by negating c; display layer re-negates the Δ row for max
    this._cOrig = Float64Array.from(c);
    const cWork = dir === 'max'
      ? Float64Array.from(c, v => -v)
      : Float64Array.from(c);

    // Count slack variables — one per ≤ / ≥ constraint, none for =
    const mSlack = signs.filter(s => s !== '=').length;
    this.mSlack = mSlack;

    // Store original problem data for canonical-form display (before any transformations)
    this._signsOrig = [...signs];
    this._bOrig     = [...b];
    this._AOrig     = [];
    for (let i = 0; i < this.m; i++) {
      const row = [];
      for (let j = 0; j < this.n; j++) row.push(A[i * this.n + j]);
      this._AOrig.push(row);
    }

    // Tableau layout:
    //   Rows  0..m-1  : constraint rows
    //   Row   m       : objective row (Δⱼ / reduced costs)
    //   Cols  0..n-1  : decision variables x₁…xₙ
    //   Cols  n..n+mSlack-1 : slack variables (only for ≤/≥ rows)
    //   Col   n+mSlack      : RHS (b)
    this.cols = this.n + mSlack + 1;
    this.bCol = this.n + mSlack;
    this.rows = this.m + 1;

    this.tableau = new Float64Array(this.rows * this.cols);
    this.basis   = new Array(this.m);

    // ── Phase 1: fill inequality constraint rows (≤ / ≥) ──────────
    const eqRows  = [];   // indices of equality constraint rows (filled in phase 2)
    let   slackIdx = 0;   // running index into the slack columns

    for (let i = 0; i < this.m; i++) {
      const sign = signs[i];
      if (sign === '=') {
        // Copy coefficients and RHS as-is; basis variable assigned in phase 2
        for (let j = 0; j < this.n; j++) {
          this._set(i, j, A[i * this.n + j]);
        }
        this._set(i, this.bCol, b[i]);
        eqRows.push(i);
      } else {
        // ≥  →  multiply by −1 so b becomes negative (primal infeasibility = dual simplex start)
        const k = sign === '>=' ? -1 : 1;
        for (let j = 0; j < this.n; j++) {
          this._set(i, j, k * A[i * this.n + j]);
        }
        const slackCol = this.n + slackIdx++;
        this._set(i, slackCol, 1);            // slack coefficient always +1 after flip
        this._set(i, this.bCol, k * b[i]);
        this.basis[i] = slackCol;
      }
    }

    // ── Phase 2: assign basic variables for equality rows ──────────
    // For each '=' row find a decision variable xⱼ whose column is a
    // unit vector for that row: coeff ≠ 0 in row i, coeff ≈ 0 in all others.
    // Normalise the row and eliminate xⱼ from every other row (including obj).
    const EPS         = DualSimplexSolver.EPSILON;
    const usedAsBasis = new Set();

    for (const i of eqRows) {
      let basicVar = -1;

      for (let j = 0; j < this.n; j++) {
        if (usedAsBasis.has(j)) continue;
        if (Math.abs(this._get(i, j)) < EPS) continue;

        // xⱼ is a candidate only if it has zero coefficient in every other row
        let clean = true;
        for (let k = 0; k < this.m; k++) {
          if (k !== i && Math.abs(this._get(k, j)) > EPS) { clean = false; break; }
        }
        if (clean) { basicVar = j; break; }
      }

      if (basicVar === -1) {
        throw new Error(
          `Рядок рівності ${i + 1}: неможливо автоматично знайти базисну змінну. ` +
          'Переконайтеся, що задача наведена в канонічній формі: ' +
          'хоча б одна змінна має ненульовий коефіцієнт лише у цьому рядку рівності.'
        );
      }

      // Normalise row so the basic variable has coefficient 1
      const pivot = this._get(i, basicVar);
      if (Math.abs(pivot - 1) > EPS) {
        for (let j = 0; j < this.cols; j++) {
          this._set(i, j, DualSimplexSolver.clean(this._get(i, j) / pivot));
        }
        this._set(i, basicVar, 1); // exact 1
      }

      this.basis[i] = basicVar;
      usedAsBasis.add(basicVar);

      // Eliminate basicVar from all other constraint rows
      for (let k = 0; k < this.m; k++) {
        if (k === i) continue;
        const factor = this._get(k, basicVar);
        if (Math.abs(factor) < EPS) continue;
        for (let j = 0; j < this.cols; j++) {
          this._set(k, j, DualSimplexSolver.clean(this._get(k, j) - factor * this._get(i, j)));
        }
        this._set(k, basicVar, 0); // exact 0
      }
    }

    // ── Phase 3: build objective row ───────────────────────────────
    // Initialise with cWork; then eliminate the reduced cost of each basic
    // decision variable (those from '=' rows) via Gauss-Jordan so that the
    // objective row correctly reflects the current basis.
    for (let j = 0; j < this.n; j++) {
      this._set(this.m, j, cWork[j]);
    }
    // (slack columns already 0; RHS starts at 0)

    for (const i of eqRows) {
      const basicVar = this.basis[i];
      const factor   = this._get(this.m, basicVar);
      if (Math.abs(factor) < EPS) continue;
      for (let j = 0; j < this.cols; j++) {
        this._set(this.m, j, DualSimplexSolver.clean(this._get(this.m, j) - factor * this._get(i, j)));
      }
      this._set(this.m, basicVar, 0); // exact 0
    }

    // History of snapshots for the interactive UI
    this.history = [];
    this._saveSnapshot();
  }


  /* ── Low-level tableau accessors ──────────── */
  _idx(row, col) { return row * this.cols + col; }
  _get(row, col) { return this.tableau[this._idx(row, col)]; }
  _set(row, col, val) { this.tableau[this._idx(row, col)] = val; }


  /* ── Snapshot (deep-copy of current state) ── */
  _saveSnapshot() {
    this.history.push({
      tableau: Float64Array.from(this.tableau),
      basis:   [...this.basis],
    });
  }

  /* Return snapshot k as a plain 2-D array (rows × cols), display-ready.
     The Δ row is stored as −cⱼ for max (max→min conversion) and as +cⱼ for
     min.  Both are returned as-is (no sign flip) so the display matches the
     canonical "Z − c·x = 0" form used in Ukrainian textbooks:
       max example: stored Δⱼ = −cⱼ → displayed as −cⱼ (≤ 0 when cⱼ ≥ 0)
       min example: stored Δⱼ = +cⱼ → displayed as +cⱼ (≥ 0 when cⱼ ≥ 0)
     The RHS column already holds the current objective value F and is shown as-is. */
  getTableau(k) {
    const snap = this.history[k];
    if (!snap) return null;
    const t = snap.tableau;
    const result = [];
    for (let i = 0; i < this.rows; i++) {
      const row = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(t[i * this.cols + j]);
      }
      result.push(row);
    }
    return result;
  }

  getBasis(k) { return [...this.history[k].basis]; }


  /* ── Optimality / feasibility checks ───────── */

  /* Dual feasibility — Δ row must have a uniform sign for non-basic columns.
     For min:  stored Δⱼ = cⱼ, dual-feasible when all ≥ 0  (standard dual simplex).
     For max:  stored Δⱼ = −cⱼ (from Z−c·x=0 form).  Two valid starting states:
       • all stored ≤ 0  (cⱼ ≥ 0, e.g. max 4x₁+5x₂) — "studfile" convention
       • all stored ≥ 0  (cⱼ ≤ 0, or equality-constraint modification) — "guide" convention
     Both represent a point where no single-direction improvement is possible.     */
  isDualFeasible(snap) {
    const t   = snap ? snap.tableau : this.tableau;
    const EPS = DualSimplexSolver.EPSILON;
    let allGe = true, allLe = true;
    for (let j = 0; j < this.bCol; j++) {
      const dj = t[this._idx(this.m, j)];
      if (dj < -EPS) allGe = false;
      if (dj >  EPS) allLe = false;
    }
    // min requires all ≥ 0; max accepts either uniform sign
    return this.dir === 'max' ? (allLe || allGe) : allGe;
  }

  /* Which dual condition is satisfied (for display messages). */
  dualFeasSign(snap) {
    const t   = snap ? snap.tableau : this.tableau;
    const EPS = DualSimplexSolver.EPSILON;
    let allLe = true;
    for (let j = 0; j < this.bCol; j++) {
      if (t[this._idx(this.m, j)] > EPS) { allLe = false; break; }
    }
    return allLe ? '≤' : '≥'; // returns '≤' if all Δⱼ ≤ 0, else '≥'
  }

  /* Primal feasibility: all bᵢ ≥ 0 */
  isPrimalFeasible(snap) {
    const t = snap ? snap.tableau : this.tableau;
    for (let i = 0; i < this.m; i++) {
      if (t[this._idx(i, this.bCol)] < -DualSimplexSolver.EPSILON) return false;
    }
    return true;
  }

  /* Both feasible ⟹ optimal */
  isOptimal() {
    return this.isPrimalFeasible(null) && this.isDualFeasible(null);
  }


  /* ── Pivot selection (dual simplex rules) ─── */

  /* Pivot row: most negative bᵢ */
  findPivotRow() {
    let pivotRow = -1;
    let mostNeg  = -DualSimplexSolver.EPSILON;
    for (let i = 0; i < this.m; i++) {
      const bi = this._get(i, this.bCol);
      if (bi < mostNeg) { mostNeg = bi; pivotRow = i; }
    }
    return pivotRow; // -1 means primal feasible (no pivot row needed)
  }

  /* Pivot column: min ratio |Δⱼ / aᵣⱼ| among aᵣⱼ < 0 */
  findPivotCol(pivotRow) {
    let pivotCol = -1;
    let minRatio = Infinity;
    for (let j = 0; j < this.bCol; j++) {
      const arj = this._get(pivotRow, j);
      if (arj < -DualSimplexSolver.EPSILON) {
        const ratio = Math.abs(this._get(this.m, j) / arj);
        if (ratio < minRatio - DualSimplexSolver.EPSILON) {
          minRatio = ratio; pivotCol = j;
        }
      }
    }
    return pivotCol; // -1 means problem is infeasible (unbounded dual)
  }


  /* ── Gauss-Jordan pivot step ─────────────── */
  _pivot(pivotRow, pivotCol) {
    const EPS = DualSimplexSolver.EPSILON;
    const clean = DualSimplexSolver.clean;

    const pivotVal = this._get(pivotRow, pivotCol);
    if (Math.abs(pivotVal) < EPS) {
      throw new Error('Pivot element is effectively zero');
    }

    // Scale pivot row; the pivot column cell becomes exactly 1
    for (let j = 0; j < this.cols; j++) {
      this._set(pivotRow, j, clean(this._get(pivotRow, j) / pivotVal));
    }
    this._set(pivotRow, pivotCol, 1); // force exact 1 — avoids x/x != 1 artifacts

    // Eliminate pivot column from all other rows (including objective row m)
    for (let i = 0; i < this.rows; i++) {
      if (i === pivotRow) continue;
      const factor = this._get(i, pivotCol);
      if (Math.abs(factor) < EPS) continue;
      for (let j = 0; j < this.cols; j++) {
        this._set(i, j, clean(this._get(i, j) - factor * this._get(pivotRow, j)));
      }
      this._set(i, pivotCol, 0); // force exact 0 — the defining result of elimination
    }

    this.basis[pivotRow] = pivotCol;
  }


  /* ── Full solve (precomputes all iterations) ─
     Returns { status: 'optimal'|'infeasible', steps }
     Each step: { pivotRow, pivotCol, tableauBefore, basisBefore }
  ─────────────────────────────────────────── */
  solve(maxIter = 50) {
    this.steps = [];

    for (let iter = 0; iter < maxIter; iter++) {
      if (this.isPrimalFeasible(null)) break; // done

      const pivotRow = this.findPivotRow();
      if (pivotRow === -1) break;

      const pivotCol = this.findPivotCol(pivotRow);
      if (pivotCol === -1) {
        this.status = 'infeasible';
        return this._buildResult();
      }

      // Record step BEFORE pivot
      const snapBefore = this.history.length - 1;
      this.steps.push({ pivotRow, pivotCol, snapIdx: snapBefore });

      this._pivot(pivotRow, pivotCol);
      this._saveSnapshot();
    }

    if (this.isPrimalFeasible(null)) {
      this.status = 'optimal';
    } else {
      // Loop exited without primal feasibility — iteration limit hit
      this.status = 'iteration_limit';
    }
    return this._buildResult();
  }

  _buildResult() {
    const result = { status: this.status, steps: this.steps, history: this.history };

    if (this.status === 'optimal') {
      // Extract solution
      const x = new Float64Array(this.n).fill(0);
      for (let i = 0; i < this.m; i++) {
        const bv = this.basis[i];
        if (bv < this.n) x[bv] = this._get(i, this.bCol);
      }
      result.x = x;

      // Objective value (re-sign for max)
      let obj = 0;
      for (let j = 0; j < this.n; j++) obj += this._cOrig[j] * x[j];
      result.objectiveValue = obj;
    }

    return result;
  }


  /* ── Validation helpers for UI ───────────────

     validatePivotRow(iterIdx, userRow)
       Returns true if userRow is the correct pivot row for iteration iterIdx.
  ─────────────────────────────────────────── */
  validatePivotRow(iterIdx, userRow) {
    return userRow === this.steps[iterIdx].pivotRow;
  }

  validatePivotCol(iterIdx, userCol) {
    return userCol === this.steps[iterIdx].pivotCol;
  }

  /*
     validateTableau(snapIdx, userGrid)
       userGrid: 2-D array [row][col] of user-entered numbers.
       Returns a same-shape boolean grid: true = correct within epsilon.
  */
  validateTableau(snapIdx, userGrid) {
    const correct = this.getTableau(snapIdx);
    return correct.map((row, i) =>
      row.map((val, j) => {
        const uv = DualSimplexSolver.parseFraction(String(userGrid[i][j]));
        if (isNaN(uv)) return false;
        return Math.abs(uv - val) <= DualSimplexSolver.EPSILON * 100 + 1e-6;
      })
    );
  }

  /*
     validateOptimalityAnswer(snapIdx, userSaysOptimal)
       snapIdx: which snapshot the student is looking at.
       userSaysOptimal: boolean — student's claim.
       Returns true if the claim matches reality.
  */
  validateOptimalityAnswer(snapIdx, userSaysOptimal) {
    const snap = this.history[snapIdx];
    const primalOk = this.isPrimalFeasible(snap);
    return userSaysOptimal === primalOk;
  }


  /* ── Parse user input: supports "3", "-1/2", "0" ── */
  static parseFraction(str) {
    str = String(str).trim();
    if (!str) return NaN;
    const parts = str.split('/');
    if (parts.length === 1) return parseFloat(str);
    if (parts.length === 2) {
      const n = parseFloat(parts[0]);
      const d = parseFloat(parts[1]);
      if (isNaN(n) || isNaN(d) || Math.abs(d) < 1e-12) return NaN;
      return n / d;
    }
    return NaN;
  }

  /* ── Utility: pretty-print a number ─────────
     Converts near-integers and common fractions to readable strings.
  ─────────────────────────────────────────── */
  static fmt(v, decimals = 4) {
    if (Math.abs(v) < DualSimplexSolver.EPSILON) return '0';
    const rounded = Math.round(v * 1e9) / 1e9;
    // Check if it's a "nice" fraction with denominator ≤ 20
    for (let d = 1; d <= 20; d++) {
      const n = Math.round(rounded * d);
      if (Math.abs(n / d - rounded) < 1e-7) {
        if (d === 1) return String(n);
        return `${n}/${d}`;
      }
    }
    return parseFloat(rounded.toFixed(decimals)).toString();
  }

  /* Variable name helpers */
  varName(colIdx) {
    if (colIdx < this.n) return `x${colIdx + 1}`;
    return `s${colIdx - this.n + 1}`;
  }
}


/* ═══════════════════════════════════════════════
   CanonicalConverter — LP problem → dual-simplex tableau

   Transforms a raw LP (c, A, b, signs, dir) into the canonical
   form required by the Dual Simplex Method and returns a ready-to-use
   matrix together with basis labels and column headers.

   Output tableau layout (identical to DualSimplexSolver internal format):
     Rows 0..m-1  : constraint rows
     Row  m       : objective row  (Δⱼ / reduced costs)
     Cols 0..n-1  : decision variables  x₁ … xₙ
     Cols n..n+m-1: slack variables     s₁ … sₘ
     Col  n+m     : RHS column          b
═══════════════════════════════════════════════ */
class CanonicalConverter {
  static EPSILON = 1e-9;

  /**
   * Convert an LP problem to the canonical dual-simplex tableau.
   *
   * Transformation rules
   * ────────────────────
   * 1. Direction
   *    max  →  min : negate every cⱼ.  The solver minimises; the caller
   *                  negates the returned objective value to recover the max.
   *    min  →  min : keep cⱼ as-is.
   *
   * 2. Constraint rows
   *    '>=' row:  multiply the entire row by −1.
   *               'a·x ≥ b'  becomes  '−a·x + sᵢ = −b'.
   *               This makes bᵢ negative → primal infeasibility,
   *               the required starting state for dual simplex.
   *    '<=' row:  keep as-is.
   *               'a·x ≤ b'  becomes   'a·x + sᵢ =  b'.
   *    '='  row:  REJECTED — equality constraints need an artificial
   *               variable or two-phase approach, which this method does
   *               not implement.
   *
   *    In both supported cases the slack coefficient is always +1
   *    (the row flip absorbs the sign for '>=' rows).
   *
   * 3. Objective row (Δⱼ)
   *    With the all-slack initial basis, Δⱼ = cⱼ (min) for decision
   *    variables and 0 for slacks.  The RHS cell starts at 0
   *    (objective value at the origin).
   *
   * 4. Dual-feasibility guard
   *    The method can only start if Δⱼ ≥ 0 for all j.  An error is
   *    thrown when this precondition is violated so the caller can
   *    surface a meaningful message rather than silently producing a
   *    wrong answer.
   *
   * @param {number[]}    c      Objective coefficients [c₁ … cₙ]
   * @param {number[][]}  A      Constraint matrix, m rows × n cols
   * @param {number[]}    b      RHS vector [b₁ … bₘ]
   * @param {string[]}    signs  Per-row inequality: '<=' | '>='
   * @param {'min'|'max'} dir    Optimisation direction
   *
   * @returns {{
   *   matrix:         number[][],   // (m+1) × (n+m+1) — last row = Δ
   *   basisVariables: string[],     // e.g. ['s1', 's2', 's3']
   *   headers:        string[]      // e.g. ['x1', 'x2', 's1', 's2', 's3', 'b']
   * }}
   *
   * @throws {Error} on invalid input or violated dual-feasibility precondition
   */
  static convert(c, A, b, signs, dir) {
    const n = c.length;
    const m = b.length;

    // ── Input validation ─────────────────────────────────────────
    if (n < 1 || m < 1) {
      throw new Error('The problem must have at least one variable and one constraint.');
    }
    if (signs.length !== m) {
      throw new Error(
        `signs array length (${signs.length}) does not match the number of constraints (${m}).`
      );
    }
    if (A.length !== m) {
      throw new Error(`Matrix A has ${A.length} rows but b has ${m} elements.`);
    }
    for (let i = 0; i < m; i++) {
      if (!Array.isArray(A[i]) || A[i].length !== n) {
        throw new Error(`Row ${i + 1} of A must have exactly ${n} elements.`);
      }
      if (signs[i] !== '<=' && signs[i] !== '>=' && signs[i] !== '=') {
        throw new Error(
          `Constraint ${i + 1} has unknown sign "${signs[i]}". Use '<=' or '>='.`
        );
      }
    }

    // ── Step 1: direction normalisation (max → min) ──────────────
    const cMin = dir === 'max' ? c.map(v => -v) : c.slice();

    // ── Step 2: build constraint rows ────────────────────────────
    const totalCols = n + m + 1;
    const bCol      = n + m;       // index of the RHS column
    const matrix    = [];

    for (let i = 0; i < m; i++) {
      const k   = signs[i] === '>=' ? -1 : 1;  // row multiplier
      const row = new Array(totalCols).fill(0);

      for (let j = 0; j < n; j++) {
        row[j] = k * A[i][j];
      }
      row[n + i] = 1;       // slack sᵢ₊₁ — always +1 after the row flip
      row[bCol]  = k * b[i];

      matrix.push(row);
    }

    // ── Step 3: objective row (Δⱼ = reduced costs) ───────────────
    const objRow = new Array(totalCols).fill(0);
    for (let j = 0; j < n; j++) {
      objRow[j] = cMin[j];
    }
    // objRow[bCol] stays 0 — current objective value at the origin
    matrix.push(objRow); // appended at index m

    // ── Step 4: dual-feasibility precondition check ───────────────
    // Dual simplex needs Δⱼ ≥ 0 for all non-basic variables initially.
    const violations = [];
    for (let j = 0; j < n; j++) {
      if (objRow[j] < -CanonicalConverter.EPSILON) {
        violations.push(`Δ${j + 1} = ${objRow[j]}`);
      }
    }
    if (violations.length > 0) {
      throw new Error(
        'The initial tableau is not dual-feasible ' +
        `(negative reduced costs: ${violations.join(', ')}). ` +
        'The Dual Simplex Method requires Δⱼ ≥ 0 for all j in the initial tableau.'
      );
    }

    // ── Step 5: labels ────────────────────────────────────────────
    const basisVariables = Array.from({ length: m }, (_, i) => `s${i + 1}`);

    const headers = [
      ...Array.from({ length: n }, (_, j) => `x${j + 1}`),
      ...Array.from({ length: m }, (_, i) => `s${i + 1}`),
      'b',
    ];

    return { matrix, basisVariables, headers };
  }
}


/* ═══════════════════════════════════════════════
   InputManager — problem form & dimension controls
═══════════════════════════════════════════════ */
class InputManager {
  constructor() {
    this.numVars = 3;
    this.numCons = 3;

    this._bindSteppers();
    this._renderForm();
    this._injectModeToggle();
    this._bindFormButtons();
  }

  _injectModeToggle() {
    if (document.getElementById('modeToggle')) return;
    const wrap = document.createElement('div');
    wrap.id = 'modeToggle';
    wrap.innerHTML =
      '<label><input type="radio" name="appMode" value="student" checked></label>' +
      '<label><input type="radio" name="appMode" value="guide"></label>';
    document.querySelector('.header-inner').appendChild(wrap);

    wrap.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const isGuide = !!wrap.querySelector('input[value="guide"]').checked;
      wrap.querySelector(`input[value="${isGuide ? 'student' : 'guide'}"]`).checked = true;
      this._showToast(isGuide ? 'Режим тренування' : 'Режим пояснення активовано');
    });
  }

  _showToast(msg) {
    const t = document.createElement('div');
    t.className = 'mode-toast';
    t.textContent = msg;
    document.body.appendChild(t);
    requestAnimationFrame(() => t.classList.add('mode-toast-show'));
    setTimeout(() => {
      t.classList.remove('mode-toast-show');
      setTimeout(() => t.remove(), 300);
    }, 2000);
  }

  _getMode() {
    const el = document.querySelector('input[name="appMode"]:checked');
    return el ? el.value : 'student';
  }

  _bindSteppers() {
    const update = (id, delta) => {
      const el = document.getElementById(id);
      const cur = parseInt(el.textContent, 10);
      const next = Math.max(2, Math.min(6, cur + delta));
      el.textContent = next;
      if (id === 'varsCount') this.numVars = next;
      else this.numCons = next;
      this._renderForm();
    };

    document.getElementById('varsDown').addEventListener('click', () => update('varsCount', -1));
    document.getElementById('varsUp'  ).addEventListener('click', () => update('varsCount',  1));
    document.getElementById('consDown').addEventListener('click', () => update('consCount', -1));
    document.getElementById('consUp'  ).addEventListener('click', () => update('consCount',  1));
  }

  _renderForm() {
    this._renderObjective();
    this._renderConstraints();
  }

  _renderObjective() {
    const list = document.getElementById('objectiveCoeffs');
    list.innerHTML = '';
    for (let j = 0; j < this.numVars; j++) {
      const term = document.createElement('div');
      term.className = 'coeff-term';
      if (j > 0) {
        const sign = document.createElement('span');
        sign.className = 'coeff-sign';
        sign.textContent = '+';
        term.appendChild(sign);
      }
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = 'any';
      inp.className = 'obj-coeff';
      inp.dataset.j = j;
      inp.placeholder = '0';
      inp.setAttribute('aria-label', `c${j + 1}`);

      const lbl = document.createElement('span');
      lbl.className = 'coeff-var';
      lbl.textContent = `x₀`.replace('₀', String.fromCharCode(0x2080 + j + 1));
      lbl.textContent = `x${j + 1}`;

      term.appendChild(inp);
      term.appendChild(lbl);
      list.appendChild(term);
    }
  }

  _renderConstraints() {
    const head = document.getElementById('constraintsHead');
    const body = document.getElementById('constraintsBody');

    head.innerHTML = '';
    body.innerHTML = '';

    // Header row
    const thRow = document.createElement('tr');
    const thEmpty = document.createElement('th');
    thEmpty.textContent = '#';
    thRow.appendChild(thEmpty);
    for (let j = 0; j < this.numVars; j++) {
      const th = document.createElement('th');
      th.textContent = `x${j + 1}`;
      thRow.appendChild(th);
    }
    const thSign = document.createElement('th'); thSign.textContent = 'Знак'; thRow.appendChild(thSign);
    const thB    = document.createElement('th'); thB.textContent = 'b';     thRow.appendChild(thB);
    head.appendChild(thRow);

    // Constraint rows
    for (let i = 0; i < this.numCons; i++) {
      const tr = document.createElement('tr');

      const tdLabel = document.createElement('td');
      tdLabel.className = 'row-label';
      tdLabel.textContent = `${i + 1}`;
      tr.appendChild(tdLabel);

      for (let j = 0; j < this.numVars; j++) {
        const td = document.createElement('td');
        const inp = document.createElement('input');
        inp.type = 'number';
        inp.step = 'any';
        inp.className = 'con-coeff';
        inp.dataset.i = i;
        inp.dataset.j = j;
        inp.placeholder = '0';
        inp.setAttribute('aria-label', `a${i + 1}${j + 1}`);
        td.appendChild(inp);
        tr.appendChild(td);
      }

      const tdSign = document.createElement('td');
      const sel = document.createElement('select');
      sel.className = 'sign-select';
      sel.dataset.i = i;
      ['≤', '≥', '='].forEach((s, idx) => {
        const opt = document.createElement('option');
        opt.value = ['<=', '>=', '='][idx];
        opt.textContent = s;
        sel.appendChild(opt);
      });
      tdSign.appendChild(sel);
      tr.appendChild(tdSign);

      const tdB = document.createElement('td');
      const inpB = document.createElement('input');
      inpB.type = 'number';
      inpB.step = 'any';
      inpB.className = 'rhs-coeff';
      inpB.dataset.i = i;
      inpB.placeholder = '0';
      inpB.setAttribute('aria-label', `b${i + 1}`);
      tdB.appendChild(inpB);
      tr.appendChild(tdB);

      body.appendChild(tr);
    }
  }

  _bindFormButtons() {
    document.getElementById('btnSolve'  ).addEventListener('click', () => this._onSolve());
    document.getElementById('btnExample').addEventListener('click', () => this._loadExample());
    document.getElementById('btnClear'  ).addEventListener('click', () => this._clearForm());

    // Inject example 2 and 3 buttons if not already present
    if (!document.getElementById('btnExample2')) {
      const btnEx = document.getElementById('btnExample');

      const btn2 = document.createElement('button');
      btn2.id        = 'btnExample2';
      btn2.className = 'btn btn-secondary';
      btn2.textContent = 'Приклад 2 (max)';
      btnEx.parentNode.insertBefore(btn2, btnEx.nextSibling);
      btn2.addEventListener('click', () => this._loadExample2());

      const btn3 = document.createElement('button');
      btn3.id        = 'btnExample3';
      btn3.className = 'btn btn-secondary';
      btn3.textContent = 'Приклад 3 (min→max)';
      btn2.parentNode.insertBefore(btn3, btn2.nextSibling);
      btn3.addEventListener('click', () => this._loadExample3());
    }
  }

  _clearForm() {
    document.querySelectorAll('.obj-coeff, .con-coeff, .rhs-coeff')
      .forEach(el => { el.value = ''; });
    document.querySelectorAll('.sign-select')
      .forEach(el => { el.value = '<='; });
    this._hideError();
  }

  _loadExample() {
    // min 2x1 + x2 + 3x3
    // s.t.  x1 + x2 + x3 >= 6
    //       x1 + 2x2      >= 8
    //      3x1 +  x2 + x3 >= 9
    document.getElementById('varsCount').textContent = '3'; this.numVars = 3;
    document.getElementById('consCount').textContent = '3'; this.numCons = 3;
    document.getElementById('direction').value = 'min';
    this._renderForm();

    const objCoeffs = [2, 1, 3];
    const aCoeffs   = [[1,1,1],[1,2,0],[3,1,1]];
    const bVals     = [6, 8, 9];
    const signs     = ['>=', '>=', '>='];

    document.querySelectorAll('.obj-coeff').forEach((el, j) => { el.value = objCoeffs[j]; });
    document.querySelectorAll('.con-coeff').forEach(el => {
      el.value = aCoeffs[el.dataset.i][el.dataset.j];
    });
    document.querySelectorAll('.rhs-coeff').forEach(el => { el.value = bVals[el.dataset.i]; });
    document.querySelectorAll('.sign-select').forEach(el => { el.value = signs[el.dataset.i]; });
    this._hideError();
  }

  _loadExample2() {
    // max Z = 4x₁ + 5x₂
    // s.t.  2x₁ + x₂ ≤ 16
    //       x₁ + 2x₂ ≥ 10  ← makes b₂ = −10 (primal infeasible row for dual simplex)
    //      3x₁ + 4x₂ ≤ 36
    // NOTE: this problem is NOT dual-feasible (Δ₁=4>0, Δ₂=5>0 for max).
    // The canonical-form card will explain why dual simplex cannot be applied.
    document.getElementById('varsCount').textContent = '2'; this.numVars = 2;
    document.getElementById('consCount').textContent = '3'; this.numCons = 3;
    document.getElementById('direction').value = 'max';
    this._renderForm();

    const objCoeffs = [4, 5];
    const aCoeffs   = [[2,1],[1,2],[3,4]];
    const bVals     = [16, 10, 36];
    const signs     = ['<=', '>=', '<='];

    document.querySelectorAll('.obj-coeff').forEach((el, j) => { el.value = objCoeffs[j]; });
    document.querySelectorAll('.con-coeff').forEach(el => {
      el.value = aCoeffs[el.dataset.i][el.dataset.j];
    });
    document.querySelectorAll('.rhs-coeff').forEach(el => { el.value = bVals[el.dataset.i]; });
    document.querySelectorAll('.sign-select').forEach(el => { el.value = signs[el.dataset.i]; });
    this._hideError();
  }

  _loadExample3() {
    // min F = 3x₁ + 2x₂ + x₃
    // s.t.  2x₁ +  x₂ + x₃ ≥ 6
    //        x₁ + 3x₂      ≥ 9
    //        x₁ +  x₂ + 2x₃ ≥ 8
    // All >= → all b become negative (primal infeasible) → dual simplex applicable.
    // min → max(−F) conversion is shown in canonical form.
    document.getElementById('varsCount').textContent = '3'; this.numVars = 3;
    document.getElementById('consCount').textContent = '3'; this.numCons = 3;
    document.getElementById('direction').value = 'min';
    this._renderForm();

    const objCoeffs = [3, 2, 1];
    const aCoeffs   = [[2,1,1],[1,3,0],[1,1,2]];
    const bVals     = [6, 9, 8];
    const signs     = ['>=', '>=', '>='];

    document.querySelectorAll('.obj-coeff').forEach((el, j) => { el.value = objCoeffs[j]; });
    document.querySelectorAll('.con-coeff').forEach(el => {
      el.value = aCoeffs[el.dataset.i][el.dataset.j];
    });
    document.querySelectorAll('.rhs-coeff').forEach(el => { el.value = bVals[el.dataset.i]; });
    document.querySelectorAll('.sign-select').forEach(el => { el.value = signs[el.dataset.i]; });
    this._hideError();
  }

  _readForm() {
    const n = this.numVars, m = this.numCons;
    const c = [], A = [], b = [], signs = [];
    let ok = true;

    document.querySelectorAll('.obj-coeff').forEach((el, j) => {
      const v = parseFloat(el.value);
      c[j] = isNaN(v) ? 0 : v;
    });

    for (let i = 0; i < m; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        const el = document.querySelector(`.con-coeff[data-i="${i}"][data-j="${j}"]`);
        const v  = parseFloat(el.value);
        if (isNaN(v)) { ok = false; break; }
        row.push(v);
      }
      A.push(row);
      const bEl = document.querySelector(`.rhs-coeff[data-i="${i}"]`);
      const bv  = parseFloat(bEl.value);
      if (isNaN(bv)) { ok = false; }
      b.push(isNaN(bv) ? 0 : bv);
      signs.push(document.querySelector(`.sign-select[data-i="${i}"]`).value);
    }

    return ok ? { n, m, c, A, b, signs } : null;
  }

  _onSolve() {
    this._hideError();
    const data = this._readForm();
    if (!data) {
      this._showError('Заповніть усі коефіцієнти матриці обмежень та вектора b.');
      return;
    }
    const dir = document.getElementById('direction').value;

    let solver;
    try {
      solver = new DualSimplexSolver(
        new Float64Array(data.c),
        new Float64Array(data.A.flat()),
        new Float64Array(data.b),
        data.signs,
        dir
      );
    } catch (e) {
      this._showError(`Помилка побудови таблиці: ${e.message}`);
      return;
    }

    // Solve only when dual-feasible; pass null to UIManager when not applicable —
    // UIManager.start() always shows the canonical-form card first and stops there
    // with an explanatory message if result is null.
    const dualOk = solver.isDualFeasible(solver.history[0]);
    const result = dualOk ? solver.solve() : null;

    const ui = new UIManager(solver, result, this._getMode());
    ui.start();
  }

  _showError(msg) {
    const banner = document.getElementById('errorBanner');
    document.getElementById('errorText').textContent = msg;
    banner.classList.remove('hidden');
  }

  _hideError() {
    document.getElementById('errorBanner').classList.add('hidden');
  }
}


/* ═══════════════════════════════════════════════
   StepAnimator — drives per-cell demo animations
═══════════════════════════════════════════════ */
class StepAnimator {
  static HL = ['hl-row','hl-col','hl-pivot','hl-neg','hl-pos','hl-warn','hl-new'];

  constructor(container) {
    this.c     = container;
    this.steps = [];
    this.idx   = -1;
    this._t    = null;
  }

  load(steps) { this.steps = steps; this.idx = -1; }

  start() { clearTimeout(this._t); this.idx = 0; this._show(); this._auto(); }
  next()  { clearTimeout(this._t); if (this.idx < this.steps.length - 1) { this.idx++; this._show(); } }
  prev()  { clearTimeout(this._t); if (this.idx > 0)                     { this.idx--; this._show(); } }

  _clear() {
    this.c.querySelectorAll('td').forEach(el =>
      StepAnimator.HL.forEach(c => el.classList.remove(c))
    );
  }

  _show() {
    const s = this.steps[this.idx]; if (!s) return;
    this._clear();
    (s.cells || []).forEach(({ sel, cls }) =>
      this.c.querySelectorAll(sel).forEach(el => el.classList.add(cls))
    );
    (s.vals || []).forEach(({ sel, text }) =>
      this.c.querySelectorAll(sel).forEach(el => { el.textContent = text; })
    );
    const te = this.c.querySelector('.demo-step-text');
    if (te && s.text != null) te.innerHTML = s.text;
    const fe = this.c.querySelector('.demo-formula');
    if (fe) {
      if (s.formula) {
        fe.textContent = s.formula;
        fe.classList.remove('hidden');
      } else {
        fe.classList.add('hidden');
      }
    }
    const ce = this.c.querySelector('.demo-counter');
    if (ce) ce.textContent = `${this.idx + 1} / ${this.steps.length}`;
  }

  _auto() {
    const s = this.steps[this.idx];
    if (!s || this.idx >= this.steps.length - 1) return;
    this._t = setTimeout(() => { this.idx++; this._show(); this._auto(); }, s.delay ?? 1800);
  }
}


/* ═══════════════════════════════════════════════
   HelpContent — rules + animated demo per stage
═══════════════════════════════════════════════ */
const HelpContent = (() => {
  function tbl(headers, rows) {
    const wrap = document.createElement('div'); wrap.className = 'demo-table-wrap';
    const t = document.createElement('table'); t.className = 'demo-table';
    const th = document.createElement('thead');
    const hr = document.createElement('tr');
    headers.forEach(h => {
      const el = document.createElement('th'); el.innerHTML = h; hr.appendChild(el);
    });
    th.appendChild(hr); t.appendChild(th);
    const tb = document.createElement('tbody');
    rows.forEach((row, ri) => {
      const tr = document.createElement('tr');
      if (row.cls) tr.className = row.cls;
      const td0 = document.createElement('td'); td0.className = 'd-basis'; td0.innerHTML = row.basis;
      tr.appendChild(td0);
      row.data.forEach((v, ci) => {
        const td = document.createElement('td');
        td.textContent = v; td.dataset.r = ri; td.dataset.c = ci;
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb); wrap.appendChild(t);
    return wrap;
  }

  const canonical = () => ({
    title: 'Побудова початкової симплекс-таблиці',
    rules: [
      'Рядок <b>≤</b>: залишаємо як є, додаємо <code>sᵢ = 1</code> у стовпець sᵢ',
      'Рядок <b>≥</b>: множимо весь рядок <em>і</em> b на <b>−1</b>, потім <code>sᵢ = 1</code>',
      'Рядок <b>Δ</b>: <code>min</code> → Δⱼ = cⱼ;&nbsp; <code>max</code> → Δⱼ = −cⱼ; для sᵢ: 0; b = 0',
      'Початковий базис — змінні <code>s₁, s₂, …, sₘ</code>, по одній у кожному рядку',
    ],
    problem: 'min  F = 2x₁ + 3x₂\n  x₁ + 2x₂ ≤ 4   (рядок ≤)\n  x₁ +  x₂ ≥ 3   (рядок ≥)',
    table: tbl(
      ['Базис','x₁','x₂','s₁','s₂','b'],
      [
        { basis:'s₁', cls:'',        data:['?','?','?','?','?'] },
        { basis:'s₂', cls:'',        data:['?','?','?','?','?'] },
        { basis:'Δ',  cls:'d-delta', data:['?','?','?','?','?'] },
      ]
    ),
    steps: [
      {
        text:'Є два рядки обмежень: перший ≤, другий ≥. Будуємо початкову симплекс-таблицю за правилами.',
        delay:2000,
      },
      {
        text:'Рядок 1 — знак <b>≤</b>: коефіцієнти залишаємо без змін, b = 4. Ставимо s₁ = 1 у своєму стовпці, s₂ = 0.',
        formula:'≤  →  x₁ + 2x₂ + s₁ = 4\n     коефіцієнти (1, 2) → без змін, b = 4, s₁ = 1',
        cells:[{ sel:'td[data-r="0"]', cls:'hl-row' }],
        vals:[ {sel:'td[data-r="0"][data-c="0"]',text:'1'}, {sel:'td[data-r="0"][data-c="1"]',text:'2'},
               {sel:'td[data-r="0"][data-c="2"]',text:'1'}, {sel:'td[data-r="0"][data-c="3"]',text:'0'},
               {sel:'td[data-r="0"][data-c="4"]',text:'4'} ],
        delay:2800,
      },
      {
        text:'Рядок 2 — знак <b>≥</b>: множимо ВЕСЬ рядок і b на −1. Потім ставимо s₂ = 1.',
        formula:'≥  →  x₁ + x₂ ≥ 3  | ×(−1)\n     −x₁ − x₂ + s₂ = −3\n     коефіцієнти (1,1)→(−1,−1), b: 3→−3',
        cells:[{ sel:'td[data-r="1"]', cls:'hl-neg' }],
        vals:[ {sel:'td[data-r="1"][data-c="0"]',text:'−1'}, {sel:'td[data-r="1"][data-c="1"]',text:'−1'},
               {sel:'td[data-r="1"][data-c="2"]',text:'0'},  {sel:'td[data-r="1"][data-c="3"]',text:'1'},
               {sel:'td[data-r="1"][data-c="4"]',text:'−3'} ],
        delay:2800,
      },
      {
        text:'Рядок Δ: напрямок <b>min</b> → Δⱼ = cⱼ. Для змінних sᵢ: Δ = 0. Стовпець b = 0 (F на початку).',
        formula:'min  →  Δ₁ = c₁ = 2\n         Δ₂ = c₂ = 3\n         Δₛ₁= Δₛ₂= 0,  b = 0',
        cells:[{ sel:'td[data-r="2"]', cls:'hl-warn' }],
        vals:[ {sel:'td[data-r="2"][data-c="0"]',text:'2'}, {sel:'td[data-r="2"][data-c="1"]',text:'3'},
               {sel:'td[data-r="2"][data-c="2"]',text:'0'}, {sel:'td[data-r="2"][data-c="3"]',text:'0'},
               {sel:'td[data-r="2"][data-c="4"]',text:'0'} ],
        delay:2500,
      },
      {
        text:'Перевірка стартових умов: b₁ = 4 ≥ 0 ✓, b₂ = −3 &lt; 0 ✗ → є від\'ємне bᵢ — стартова умова двоїстого методу.',
        formula:'b₁ = 4 ≥ 0  → рядок допустимий\nb₂ = −3 < 0 → рядок недопустимий ← стартова умова',
        cells:[
          {sel:'td[data-r="0"][data-c="4"]', cls:'hl-pos'},
          {sel:'td[data-r="1"][data-c="4"]', cls:'hl-neg'},
        ],
        delay:2500,
      },
      {
        text:'✓ Таблиця побудована. Подвійна допустимість: Δ₁=2≥0, Δ₂=3≥0 → метод застосовний.',
        formula:'Δⱼ ≥ 0 для всіх j → подвійна допустимість ✓\nb₂ < 0           → прімальна недопустимість ✓\nДвоїстий симплекс-метод можна застосовувати!',
        cells:[
          {sel:'td[data-r="0"]', cls:'hl-pos'}, {sel:'td[data-r="1"]', cls:'hl-neg'}, {sel:'td[data-r="2"]', cls:'hl-warn'},
        ],
      },
    ],
  });

  const optimality = () => ({
    title: 'Перевірка оптимальності плану',
    rules: [
      'Перевіряємо <b>лише стовпець b</b> (права частина обмежень)',
      'Якщо <b>всі bᵢ ≥ 0</b> → план прімально допустимий → <b>оптимальний</b>',
      'Якщо <b>хоча б одне bᵢ &lt; 0</b> → план недопустимий → виконуємо ітерацію',
    ],
    problem: 'Чи є поточний план оптимальним?',
    table: tbl(
      ['Базис','x₁','x₂','s₁','s₂','b'],
      [
        { basis:'s₁', cls:'',        data:['1', '2', '1','0', '2'] },
        { basis:'s₂', cls:'',        data:['−1','−1','0','1','−3'] },
        { basis:'Δ',  cls:'d-delta', data:['2', '3', '0','0', '0'] },
      ]
    ),
    steps: [
      {
        text:'Оптимальність перевіряється за стовпцем <b>b</b>. Потрібно щоб усі bᵢ ≥ 0.',
        formula:'Критерій оптимальності:\n  всі bᵢ ≥ 0  →  план оптимальний\n  є bᵢ < 0   →  потрібна ітерація',
        delay:2000,
      },
      {
        text:'b₁ = 2 ≥ 0 ✓ — рядок 1 допустимий.',
        formula:'b₁ = 2 ≥ 0  ✓',
        cells:[{sel:'td[data-r="0"][data-c="4"]', cls:'hl-pos'}],
        delay:1800,
      },
      {
        text:'b₂ = −3 &lt; 0 ✗ — знайдено від\'ємне! Оптимальності немає.',
        formula:'b₂ = −3 < 0  ✗  ← порушення!',
        cells:[
          {sel:'td[data-r="0"][data-c="4"]', cls:'hl-pos'},
          {sel:'td[data-r="1"][data-c="4"]', cls:'hl-neg'},
        ],
        delay:1800,
      },
      {
        text:'Є від\'ємні bᵢ → план прімально <b>недопустимий</b> → виконуємо ітерацію.',
        cells:[
          {sel:'td[data-r="0"][data-c="4"]', cls:'hl-pos'},
          {sel:'td[data-r="1"][data-c="4"]', cls:'hl-neg'},
        ],
        delay:1500,
      },
      {
        text:'→ Відповідь: план <b>НЕ оптимальний</b>. Переходимо до вибору ведучого рядка.',
        formula:'∃ bᵢ < 0  →  "Ні, потрібна ітерація"',
        cells:[{sel:'td[data-r="1"]', cls:'hl-neg'}],
      },
    ],
  });

  const pivotRow = () => ({
    title: 'Вибір ведучого рядка',
    rules: [
      'Знайдіть усі рядки де <b>bᵢ &lt; 0</b>',
      'Ведучий рядок — той де <b>bᵢ найменше (найбільш від\'ємне)</b>',
      'Якщо кілька рядків мають однакове мінімальне b — оберіть будь-який',
    ],
    problem: 'Оберіть ведучий рядок (найбільш від\'ємне b):',
    table: tbl(
      ['Базис','x₁','x₂','s₁','b'],
      [
        { basis:'s₁', cls:'',        data:['1', '2','1', '3'] },
        { basis:'s₂', cls:'',        data:['−2','1','0','−5'] },
        { basis:'s₃', cls:'',        data:['1','−1','0','−1'] },
        { basis:'Δ',  cls:'d-delta', data:['3', '4','0', '0'] },
      ]
    ),
    steps: [
      {
        text:'Шукаємо ведучий рядок. Правило: рядок з <b>найбільш від\'ємним bᵢ</b>.',
        formula:'Ведучий рядок r: bᵣ = min{ bᵢ : bᵢ < 0 }',
        delay:2000,
      },
      {
        text:'b₁ = 3 ≥ 0 — цей рядок не підходить (bᵢ невід\'ємне).',
        formula:'b₁ = 3 ≥ 0  → ігноруємо',
        cells:[{sel:'td[data-r="0"][data-c="3"]', cls:'hl-pos'}],
        delay:1800,
      },
      {
        text:'b₂ = −5 &lt; 0 — перший кандидат.',
        formula:'b₂ = −5 < 0  → кандидат',
        cells:[
          {sel:'td[data-r="0"][data-c="3"]', cls:'hl-pos'},
          {sel:'td[data-r="1"][data-c="3"]', cls:'hl-neg'},
        ],
        delay:1800,
      },
      {
        text:'b₃ = −1 &lt; 0 — теж від\'ємне, але −1 &gt; −5.',
        formula:'b₃ = −1 < 0  → кандидат\nАле: −5 < −1  → рядок 2 "гірше"',
        cells:[
          {sel:'td[data-r="0"][data-c="3"]', cls:'hl-pos'},
          {sel:'td[data-r="1"][data-c="3"]', cls:'hl-neg'},
          {sel:'td[data-r="2"][data-c="3"]', cls:'hl-warn'},
        ],
        delay:1800,
      },
      {
        text:'Порівнюємо: min(−5, −1) = −5 → обираємо рядок 2.',
        formula:'min(b₂, b₃) = min(−5, −1) = −5\n→ ведучий рядок = рядок 2',
        cells:[
          {sel:'td[data-r="0"][data-c="3"]', cls:'hl-pos'},
          {sel:'td[data-r="1"][data-c="3"]', cls:'hl-neg'},
          {sel:'td[data-r="2"][data-c="3"]', cls:'hl-warn'},
          {sel:'td[data-r="1"]', cls:'hl-row'},
        ],
        delay:2000,
      },
      {
        text:'→ <b>Ведучий рядок = рядок 2</b> (найбільш від\'ємне b₂ = −5).',
        cells:[{sel:'td[data-r="1"]', cls:'hl-neg'}],
      },
    ],
  });

  const pivotCol = () => ({
    title: 'Вибір ведучого стовпця',
    rules: [
      'Беремо лише елементи ведучого рядка де <b>aᵣⱼ &lt; 0</b> — інші ігноруємо',
      'Для кожного такого j рахуємо: <code>|Δⱼ / aᵣⱼ|</code>',
      'Ведучий стовпець — той j де відношення <b>мінімальне</b>',
    ],
    problem: 'Ведучий рядок: рядок 2 (виділено). Оберіть ведучий стовпець:',
    table: tbl(
      ['Базис','x₁','x₂','s₁','b'],
      [
        { basis:'s₁', cls:'',        data:['1', '2', '1','3'] },
        { basis:'s₂', cls:'',        data:['−2','1','−1','−5'] },
        { basis:'Δ',  cls:'d-delta', data:['4', '0', '3', '0'] },
      ]
    ),
    steps: [
      {
        text:'Ведучий рядок — рядок 2 (виділено). Тепер вибираємо ведучий стовпець.',
        formula:'Ведучий стовпець s:\n  мін{ |Δⱼ/aᵣⱼ| : aᵣⱼ < 0 }',
        cells:[{sel:'td[data-r="1"]', cls:'hl-row'}],
        delay:2000,
      },
      {
        text:'Шукаємо <b>від\'ємні елементи</b> у ведучому рядку: a₂₁ = −2 &lt; 0 ✓, a₂₂ = 1 ≥ 0 — ігноруємо, a₂₃ = −1 &lt; 0 ✓.',
        formula:'a₂₁ = −2 < 0  → кандидат\na₂₂ =  1 ≥ 0  → ігноруємо\na₂₃ = −1 < 0  → кандидат',
        cells:[
          {sel:'td[data-r="1"][data-c="0"]', cls:'hl-neg'},
          {sel:'td[data-r="1"][data-c="2"]', cls:'hl-neg'},
        ],
        delay:2800,
      },
      {
        text:'Відношення для x₁ (j=1): |Δ₁ / a₂₁|',
        formula:'|Δ₁ / a₂₁| = |4 / (−2)| = 4/2 = 2',
        cells:[
          {sel:'td[data-r="1"][data-c="0"]', cls:'hl-neg'},
          {sel:'td[data-r="2"][data-c="0"]', cls:'hl-warn'},
        ],
        delay:2200,
      },
      {
        text:'Відношення для s₁ (j=3): |Δ₃ / a₂₃|',
        formula:'|Δ₃ / a₂₃| = |3 / (−1)| = 3/1 = 3',
        cells:[
          {sel:'td[data-r="1"][data-c="2"]', cls:'hl-neg'},
          {sel:'td[data-r="2"][data-c="2"]', cls:'hl-warn'},
        ],
        delay:2200,
      },
      {
        text:'Порівнюємо: 2 &lt; 3 → мінімум у стовпці x₁.',
        formula:'min(2, 3) = 2  →  ведучий стовпець = x₁',
        cells:[
          {sel:'td[data-r="0"][data-c="0"]', cls:'hl-col'},
          {sel:'td[data-r="1"][data-c="0"]', cls:'hl-neg'},
          {sel:'td[data-r="2"][data-c="0"]', cls:'hl-col'},
        ],
        delay:2200,
      },
      {
        text:'→ <b>Ведучий стовпець = x₁</b>. Ведучий елемент a₂₁ = −2.',
        cells:[
          {sel:'td[data-r="0"][data-c="0"]', cls:'hl-col'},
          {sel:'td[data-r="1"][data-c="0"]', cls:'hl-pivot'},
          {sel:'td[data-r="2"][data-c="0"]', cls:'hl-col'},
        ],
      },
    ],
  });

  const gauss = () => ({
    title: 'Крок Гаусса-Жордана',
    rules: [
      '<b>Ведучий рядок</b>: ділимо кожен елемент на ведучий елемент <code>aᵣₛ</code>',
      '<b>Кожен інший рядок i</b>: новий_рядокᵢ = старий_рядокᵢ − <code>aᵢₛ</code> × новий_ведучий_рядок',
      'Після перетворення: ведучий елемент = <b>1</b>, решта елементів стовпця = <b>0</b>',
      'Рядок Δ обробляється <b>так само</b>, як інші рядки',
    ],
    problem: 'Ведучий рядок: 2, стовпець: x₁. Ведучий елемент a₂₁ = −2.',
    table: tbl(
      ['Базис','x₁','x₂','s₁','s₂','b'],
      [
        { basis:'s₁', cls:'',        data:['1', '2','1','0', '4'] },
        { basis:'s₂', cls:'',        data:['−2','1','0','1','−6'] },
        { basis:'Δ',  cls:'d-delta', data:['4', '3','0','0', '0'] },
      ]
    ),
    steps: [
      {
        text:'Ведучий елемент a₂₁ = −2 — на перетині ведучого рядка (рядок 2) та стовпця (x₁).',
        formula:'Ведучий елемент: a₂₁ = −2',
        cells:[
          {sel:'td[data-r="1"]',             cls:'hl-row'},
          {sel:'td[data-r="0"][data-c="0"]', cls:'hl-col'},
          {sel:'td[data-r="2"][data-c="0"]', cls:'hl-col'},
          {sel:'td[data-r="1"][data-c="0"]', cls:'hl-pivot'},
        ],
        delay:2500,
      },
      {
        text:'<b>Крок 1: ведучий рядок ÷ ведучий елемент</b>. Ділимо кожен елемент рядка 2 на −2.',
        formula:'Рядок 2 ÷ (−2):\n[−2, 1, 0, 1, −6] ÷ (−2)',
        cells:[{sel:'td[data-r="1"]', cls:'hl-row'}],
        delay:2500,
      },
      {
        text:'Новий ведучий рядок (рядок 2):',
        formula:'[−2÷(−2), 1÷(−2), 0÷(−2), 1÷(−2), −6÷(−2)]\n= [  1,    −½,      0,    −½,      3  ]',
        cells:[{sel:'td[data-r="1"]', cls:'hl-new'}],
        vals:[
          {sel:'td[data-r="1"][data-c="0"]',text:'1'},   {sel:'td[data-r="1"][data-c="1"]',text:'−1/2'},
          {sel:'td[data-r="1"][data-c="2"]',text:'0'},   {sel:'td[data-r="1"][data-c="3"]',text:'−1/2'},
          {sel:'td[data-r="1"][data-c="4"]',text:'3'},
        ],
        delay:2800,
      },
      {
        text:'<b>Крок 2: інші рядки.</b> Рядок 1: коефіцієнт = a₁₁ = 1 (зі старої таблиці).',
        formula:'Новий рядок 1 = Старий рядок 1 − 1 × Новий рядок 2\n= [1, 2, 1, 0, 4] − 1×[1, −½, 0, −½, 3]',
        cells:[
          {sel:'td[data-r="0"]', cls:'hl-calc'},
          {sel:'td[data-r="1"]', cls:'hl-row'},
        ],
        delay:2800,
      },
      {
        text:'Новий рядок 1:',
        formula:'[1−1·1, 2−1·(−½), 1−1·0, 0−1·(−½), 4−1·3]\n= [  0,    5/2,      1,    1/2,      1  ]',
        cells:[{sel:'td[data-r="0"]', cls:'hl-new'}],
        vals:[
          {sel:'td[data-r="0"][data-c="0"]',text:'0'},   {sel:'td[data-r="0"][data-c="1"]',text:'5/2'},
          {sel:'td[data-r="0"][data-c="2"]',text:'1'},   {sel:'td[data-r="0"][data-c="3"]',text:'1/2'},
          {sel:'td[data-r="0"][data-c="4"]',text:'1'},
        ],
        delay:2800,
      },
      {
        text:'<b>Крок 3: рядок Δ.</b> Коефіцієнт = Δ₁ = 4 (зі старої таблиці).',
        formula:'Новий рядок Δ = Старий рядок Δ − 4 × Новий рядок 2\n= [4, 3, 0, 0, 0] − 4×[1, −½, 0, −½, 3]',
        cells:[
          {sel:'td[data-r="2"]', cls:'hl-calc'},
          {sel:'td[data-r="1"]', cls:'hl-row'},
        ],
        delay:2800,
      },
      {
        text:'Новий рядок Δ:',
        formula:'[4−4·1, 3−4·(−½), 0−4·0, 0−4·(−½), 0−4·3]\n= [  0,     5,        0,     2,       −12 ]',
        cells:[{sel:'td[data-r="2"]', cls:'hl-new'}],
        vals:[
          {sel:'td[data-r="2"][data-c="0"]',text:'0'}, {sel:'td[data-r="2"][data-c="1"]',text:'5'},
          {sel:'td[data-r="2"][data-c="2"]',text:'0'}, {sel:'td[data-r="2"][data-c="3"]',text:'2'},
          {sel:'td[data-r="2"][data-c="4"]',text:'−12'},
        ],
        delay:2800,
      },
      {
        text:'✓ Крок Гаусса-Жордана завершено! Стовпець x₁ тепер одиничний: 0, 1, 0.',
        formula:'Стовпець x₁ після перетворення: [0, 1, 0]\nБазисна змінна рядка 2 змінилась: s₂ → x₁',
        cells:[
          {sel:'td[data-r="0"]', cls:'hl-pos'},
          {sel:'td[data-r="1"]', cls:'hl-pos'},
          {sel:'td[data-r="2"]', cls:'hl-pos'},
        ],
      },
    ],
  });

  return { canonical, optimality, pivotRow, pivotCol, gauss };
})();


/* ═══════════════════════════════════════════════
   UIManager — step-by-step interactive flow
═══════════════════════════════════════════════ */
class UIManager {
  constructor(solver, result, mode = 'student') {
    this.solver  = solver;
    this.result  = result;
    this.mode    = mode;
    this.stepIdx = 0;

    this.output  = document.getElementById('solutionOutput');
    this.wrapper = document.getElementById('solutionWrapper');
    this.banner  = document.getElementById('resultBanner');
  }

  start() {
    this.output.innerHTML = '';
    this.wrapper.classList.remove('hidden');
    this.banner.classList.add('hidden');
    this.banner.className = 'result-banner hidden';
    window.scrollTo({ top: this.wrapper.offsetTop - 80, behavior: 'smooth' });

    this._renderCanonicalForm();

    if (!this.result) return; // not dual-feasible — canonical card already explains why

    if (this.mode === 'guide') {
      this._startGuide();
    }
    // student mode: _renderCanonicalForm() triggers _renderStep(0) after canonical form is verified
  }

  /* ══════════════════════════════════════════
     CANONICAL FORM — always rendered first
  ══════════════════════════════════════════ */
  _renderCanonicalForm() {
    const solver  = this.solver;
    const { n, m, dir, mSlack } = solver;
    // Dual-feasibility condition depends on which convention the problem uses.
    // Determined dynamically from the actual Δ-row sign.
    const snap0    = solver.history[0];
    const dualSign = solver.dualFeasSign(snap0); // '≤' or '≥'
    const dualCond = `Δⱼ ${dualSign} 0`;

    const { card, body } = this._makeCard('К', 'Канонічна форма задачі');
    this.output.appendChild(card);

    // ── Helper: format a linear expression as HTML ──────────────
    const varNames   = Array.from({ length: n },      (_, j) => `x<sub>${j+1}</sub>`);
    const slackNames = Array.from({ length: mSlack },  (_, k) => `s<sub>${k+1}</sub>`);
    const allNames   = [...varNames, ...slackNames];

    const fmtExpr = (coeffs, names) => {
      let s = ''; let first = true;
      for (let j = 0; j < coeffs.length; j++) {
        const c = coeffs[j];
        if (Math.abs(c) < DualSimplexSolver.EPSILON) continue;
        const abs = DualSimplexSolver.fmt(Math.abs(c));
        const coeff = Math.abs(c) === 1 ? '' : abs;
        if (first) { s += (c < 0 ? '−' : '') + coeff + names[j]; first = false; }
        else        s += (c < 0 ? ' − ' : ' + ') + coeff + names[j];
      }
      return s || '0';
    };

    // ── Phase 1: original formulation ───────────────────────────
    const p1 = document.createElement('div'); p1.className = 'phase';
    const p1q = document.createElement('p'); p1q.className = 'phase-question';
    p1q.textContent = 'Вихідна задача:';
    p1.appendChild(p1q);

    const objDiv = document.createElement('div'); objDiv.className = 'step-note';
    objDiv.innerHTML = `<strong>F = ${fmtExpr([...solver._cOrig], varNames)} → ${dir}</strong>`;
    p1.appendChild(objDiv);

    const conDiv = document.createElement('div'); conDiv.className = 'step-note';
    conDiv.innerHTML = solver._AOrig.map((row, i) => {
      const sc = solver._signsOrig[i] === '>=' ? '≥' : solver._signsOrig[i] === '<=' ? '≤' : '=';
      return `${fmtExpr(row, varNames)} ${sc} ${DualSimplexSolver.fmt(solver._bOrig[i])}`;
    }).join('<br>');
    p1.appendChild(conDiv);

    if (dir === 'min') {
      const convertDiv = document.createElement('div'); convertDiv.className = 'step-note';
      const negC = [...solver._cOrig].map(v => -v);
      convertDiv.innerHTML =
        `<strong>Перетворення напрямку:</strong> min F → max(−F)<br>` +
        `<strong>F' = ${fmtExpr(negC, varNames)} → max</strong>`;
      p1.appendChild(convertDiv);
    }

    body.appendChild(p1);

    // ── Phase 2: transformation table ───────────────────────────
    const p2 = document.createElement('div'); p2.className = 'phase';
    const p2q = document.createElement('p'); p2q.className = 'phase-question';
    p2q.textContent = 'Перетворення до канонічного вигляду:';
    p2.appendChild(p2q);

    const tbl = document.createElement('table'); tbl.className = 'simplex-table';
    const thd = document.createElement('thead');
    const thr = document.createElement('tr');
    ['#', 'Знак', 'Вихідний рядок', 'Дія', 'Канонічний рядок'].forEach(txt => {
      const th = document.createElement('th'); th.innerHTML = txt; thr.appendChild(th);
    });
    thd.appendChild(thr); tbl.appendChild(thd);

    const tbdy = document.createElement('tbody');
    let sIdx = 0;
    const initTab = solver.getTableau(0);
    const deltaRow = initTab[m];

    for (let i = 0; i < m; i++) {
      const sign = solver._signsOrig[i];
      const aRow = solver._AOrig[i];
      const bi   = solver._bOrig[i];
      const tr   = document.createElement('tr');
      const sc   = sign === '>=' ? '≥' : sign === '<=' ? '≤' : '=';

      const tdN = document.createElement('td'); tdN.textContent = i + 1;
      const tdS = document.createElement('td'); tdS.innerHTML = `<code>${sc}</code>`;
      const tdO = document.createElement('td');
      tdO.innerHTML = `${fmtExpr(aRow, varNames)} ${sc} ${DualSimplexSolver.fmt(bi)}`;
      const tdA = document.createElement('td');
      const tdC = document.createElement('td');

      if (sign === '<=') {
        tdA.innerHTML = `+ s<sub>${sIdx+1}</sub> ≥ 0`;
        const row = [...aRow, ...Array(mSlack).fill(0)];
        row[n + sIdx] = 1;
        tdC.innerHTML = `${fmtExpr(row, allNames)} = ${DualSimplexSolver.fmt(bi)}`;
        sIdx++;
      } else if (sign === '>=') {
        tdA.innerHTML = `×(−1), + s<sub>${sIdx+1}</sub> ≥ 0`;
        const row = [...aRow.map(v => -v), ...Array(mSlack).fill(0)];
        row[n + sIdx] = 1;
        tdC.innerHTML = `${fmtExpr(row, allNames)} = ${DualSimplexSolver.fmt(-bi)}`;
        sIdx++;
      } else { // '='
        const bv = solver.basis[i];
        tdA.innerHTML = `x<sub>${bv+1}</sub> — базисна`;
        tdC.innerHTML = `${fmtExpr(aRow, varNames)} = ${DualSimplexSolver.fmt(bi)}`;
      }

      [tdN, tdS, tdO, tdA, tdC].forEach(td => tr.appendChild(td));
      tbdy.appendChild(tr);
    }

    // Objective row in the table
    const objTr = document.createElement('tr'); objTr.className = 'row-delta';
    const tdON  = document.createElement('td'); tdON.textContent = 'Δ';
    const tdOS  = document.createElement('td');
    tdOS.innerHTML = dir === 'max' ? 'max' : 'min';
    const tdOO  = document.createElement('td');
    tdOO.innerHTML = `F = ${fmtExpr([...solver._cOrig], varNames)} → ${dir}`;
    const tdOA  = document.createElement('td');
    // For max: stored Δⱼ = −cⱼ (from Z − c·x = 0).
    // For min: stored Δⱼ = +cⱼ.  Condition for dual simplex: ${dualCond}.
    tdOA.innerHTML = dir === 'max'
      ? `Δ<sub>j</sub> = −c<sub>j</sub> &nbsp;(${dualCond})`
      : `min→max(−F): Δ<sub>j</sub> = −c′<sub>j</sub> = c<sub>j</sub> &nbsp;(${dualCond})`;
    const tdOC  = document.createElement('td');
    const dCoeffs = [...deltaRow.slice(0, n + mSlack)];
    tdOC.innerHTML = `${fmtExpr(dCoeffs, allNames)} &nbsp;[F = ${DualSimplexSolver.fmt(deltaRow[solver.bCol])}]`;
    [tdON, tdOS, tdOO, tdOA, tdOC].forEach(td => objTr.appendChild(td));
    tbdy.appendChild(objTr);
    tbl.appendChild(tbdy);
    p2.appendChild(tbl);
    if (this.mode === 'student') p2.classList.add('hidden');
    body.appendChild(p2);

    // ── Shared helper: build the feasibility-summary phase ──────
    const buildP4 = () => {
      const el = document.createElement('div'); el.className = 'phase';
      const dualOk       = solver.isDualFeasible(snap0);
      const primalInfeas = !solver.isPrimalFeasible(snap0);

      const dualDiv = document.createElement('div');
      dualDiv.className = `phase-feedback ${dualOk ? 'success' : 'error'}`;
      dualDiv.innerHTML = dualOk
        ? `✓ Подвійна допустимість: виконується (всі ${dualCond}) — метод застосовний`
        : `✕ Подвійна допустимість: <strong>порушена</strong> — не всі ${dualCond}.<br>` +
          `Двоїстий симплекс-метод вимагає ${dualCond} для всіх небазисних змінних ` +
          `в початковій таблиці. Змініть задачу або оберіть інший метод.`;
      el.appendChild(dualDiv);

      const primDiv = document.createElement('div');
      primDiv.className = `phase-feedback ${primalInfeas ? 'success' : 'info'}`;
      primDiv.innerHTML = primalInfeas
        ? '✓ Є від\'ємні bᵢ — план первинно недопустимий (стартова умова двоїстого методу)'
        : 'ℹ Всі bᵢ ≥ 0 — план вже первинно допустимий (ітерацій не потрібно, розв\'язок знайдено одразу)';
      el.appendChild(primDiv);
      return el;
    };

    // ── Phase 3: initial simplex tableau ────────────────────────
    if (this.mode !== 'student') {
      const p3 = document.createElement('div'); p3.className = 'phase';
      const p3q = document.createElement('p'); p3q.className = 'phase-question';
      p3q.textContent = 'Початкова симплекс-таблиця:';
      p3.appendChild(p3q);
      p3.appendChild(this._buildTable(0).wrap);
      body.appendChild(p3);
      body.appendChild(buildP4());
      return;
    }

    // ── Student mode: student fills in the canonical tableau ─────
    const p3 = document.createElement('div'); p3.className = 'phase';
    const p3q = document.createElement('p'); p3q.className = 'phase-question';
    p3q.textContent = 'Заповніть початкову симплекс-таблицю (канонічна форма):';
    p3.appendChild(p3q);

    p3.appendChild(this._buildHelpPanel('canonical'));

    const { wrap: editWrap, inputCells } = this._buildFullInputTable(0);
    p3.appendChild(editWrap);

    const verifyBar = document.createElement('div');
    verifyBar.className = 'recalc-controls';
    const btnVerify = document.createElement('button');
    btnVerify.className = 'btn btn-primary';
    btnVerify.textContent = 'Перевірити';
    verifyBar.appendChild(btnVerify);
    p3.appendChild(verifyBar);

    const verifyFb = document.createElement('div');
    verifyFb.className = 'phase-feedback hidden';
    p3.appendChild(verifyFb);
    body.appendChild(p3);

    const p4 = buildP4();
    p4.classList.add('hidden');
    body.appendChild(p4);

    let verifyMisses = 0;

    btnVerify.addEventListener('click', () => {
      let allOk = true;
      inputCells.forEach(({ inp, correctVal }) => {
        inp.classList.remove('correct', 'wrong');
        const uv = DualSimplexSolver.parseFraction(inp.value);
        const ok = !isNaN(uv) && Math.abs(uv - correctVal) <= 1e-6;
        inp.classList.add(ok ? 'correct' : 'wrong');
        if (!ok) allOk = false;
      });

      if (allOk) {
        verifyFb.className = 'phase-feedback success';
        verifyFb.textContent = 'Правильно! Канонічну форму побудовано вірно.';
        btnVerify.disabled = true;
        p2.classList.remove('hidden');
        p4.classList.remove('hidden');
        p4.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        if (this.result) {
          setTimeout(() => this._renderStep(0), 800);
        }
      } else {
        verifyMisses++;
        let msg = 'Деякі значення невірні (червоний колір).';
        if (verifyMisses === 1) {
          msg += ' Підказка: ≥ → рядок і b на −1; ≤ → без змін; sᵢ = 1 у своєму стовпці. Скористайтесь панеллю «Правила та приклад» вище.';
        } else {
          msg += ' Перевірте знаки коефіцієнтів, значення b та рядок Δ.';
        }
        verifyFb.className = 'phase-feedback error';
        verifyFb.textContent = msg;
      }
    });
  }

  /* ══════════════════════════════════════════
     GUIDE MODE — renders all steps at once
  ══════════════════════════════════════════ */
  _startGuide() {
    const steps = this.result.steps;
    for (let i = 0; i <= steps.length; i++) {
      this._renderGuideStep(i);
    }
    this._showFinalResult();
  }

  _renderGuideStep(stepIdx) {
    const solver  = this.solver;
    const steps   = this.result.steps;
    const isLast  = stepIdx >= steps.length;
    const snapIdx = isLast ? solver.history.length - 1 : steps[stepIdx].snapIdx;
    const cardNum = stepIdx + 1;

    const pivotRow = isLast ? -1 : steps[stepIdx].pivotRow;
    const pivotCol = isLast ? -1 : steps[stepIdx].pivotCol;

    const title = isLast ? 'Оптимальний план досягнуто' : `Ітерація ${cardNum}`;
    const { card, body } = this._makeCard(cardNum, title);
    this.output.appendChild(card);

    const tab   = solver.getTableau(snapIdx);
    const basis = solver.getBasis(snapIdx);

    // ── Current table (with pivot highlighted if not last) ──
    const tablePhase = document.createElement('div');
    tablePhase.className = 'phase';
    const tLabel = document.createElement('p');
    tLabel.className = 'phase-question';
    tLabel.textContent = isLast ? 'Фінальна симплекс-таблиця:' : 'Поточна симплекс-таблиця:';
    tablePhase.appendChild(tLabel);
    tablePhase.appendChild(this._buildTable(snapIdx, pivotRow, pivotCol).wrap);
    body.appendChild(tablePhase);

    if (isLast) {
      const note = document.createElement('div');
      note.className = 'step-note';
      note.innerHTML = '<strong>Перевірка:</strong> всі bᵢ ≥ 0 — план прімально допустимий і оптимальний.';
      const notePhase = document.createElement('div');
      notePhase.className = 'phase';
      notePhase.appendChild(note);
      body.appendChild(notePhase);
      return;
    }

    // ── Explanation notes ──
    const explPhase = document.createElement('div');
    explPhase.className = 'phase';

    // Optimality check
    const negB = [];
    for (let i = 0; i < solver.m; i++) {
      const bi = tab[i][solver.bCol];
      if (bi < -DualSimplexSolver.EPSILON) negB.push({ i, bi });
    }
    const note1 = document.createElement('div');
    note1.className = 'step-note';
    note1.innerHTML =
      `<strong>Перевірка оптимальності:</strong> є від'ємні bᵢ: ` +
      negB.map(({ i, bi }) => `b<sub>${i+1}</sub> = ${DualSimplexSolver.fmt(bi)}`).join(', ') +
      '. Виконуємо ітерацію двоїстого симплекс-методу.';
    explPhase.appendChild(note1);

    // Pivot row
    const note2 = document.createElement('div');
    note2.className = 'step-note';
    note2.innerHTML =
      `<strong>Ведучий рядок:</strong> рядок ${pivotRow + 1} ` +
      `(базисна змінна ${solver.varName(basis[pivotRow])}) — ` +
      `найбільш від'ємне b<sub>${pivotRow+1}</sub> = ${DualSimplexSolver.fmt(tab[pivotRow][solver.bCol])}.`;
    explPhase.appendChild(note2);

    // Pivot column: show all ratios
    const ratios = [];
    for (let j = 0; j < solver.bCol; j++) {
      const arj = tab[pivotRow][j];
      const dj  = tab[solver.m][j];
      if (arj < -DualSimplexSolver.EPSILON) {
        ratios.push({
          j, arj, dj,
          ratio: Math.abs(dj / arj),
        });
      }
    }
    const ratioStr = ratios.map(({ j, arj, dj, ratio }) =>
      `|Δ<sub>${j+1}</sub>/a<sub>${pivotRow+1},${j+1}</sub>| = ` +
      `|${DualSimplexSolver.fmt(dj)}/${DualSimplexSolver.fmt(arj)}| = ${DualSimplexSolver.fmt(ratio)}`
    ).join('&nbsp;&nbsp;|&nbsp;&nbsp;');
    const note3 = document.createElement('div');
    note3.className = 'step-note';
    note3.innerHTML =
      `<strong>Ведучий стовпець:</strong> відношення для a<sub>rj</sub> &lt; 0: ${ratioStr}. ` +
      `Мінімум → стовпець <em>${solver.varName(pivotCol)}</em> (j = ${pivotCol + 1}).`;
    explPhase.appendChild(note3);

    // Gauss-Jordan formula reminder
    const note4 = document.createElement('div');
    note4.className = 'step-note';
    note4.innerHTML =
      `<strong>Крок Гаусса-Жордана:</strong> ` +
      `ведучий рядок ділимо на елемент a<sub>${pivotRow+1},${pivotCol+1}</sub> = ` +
      `${DualSimplexSolver.fmt(tab[pivotRow][pivotCol])}. ` +
      `Для решти рядків i: нов. рядок<sub>i</sub> = стар. рядок<sub>i</sub> − ` +
      `a<sub>i,${pivotCol+1}</sub> × (новий ведучий рядок).`;
    explPhase.appendChild(note4);

    body.appendChild(explPhase);

    // ── Result table after pivot ──
    const nextPhase = document.createElement('div');
    nextPhase.className = 'phase';
    const nLabel = document.createElement('p');
    nLabel.className = 'phase-question';
    nLabel.textContent = 'Таблиця після виконання кроку Гаусса-Жордана:';
    nextPhase.appendChild(nLabel);
    nextPhase.appendChild(this._buildTable(snapIdx + 1).wrap);
    body.appendChild(nextPhase);
  }

  /* ── Column header labels ─── */
  _colHeaders() {
    const { n, mSlack } = this.solver;
    const hdrs = [];
    for (let j = 0; j < n; j++) hdrs.push(`x${j + 1}`);
    for (let k = 0; k < mSlack; k++) hdrs.push(`s${k + 1}`);
    hdrs.push('b');
    return hdrs;
  }

  /* ── Build a read-only simplex table DOM element ──
     pivotRow / pivotCol: highlight indices (-1 = none)
  ─────────────────────────────────────────────── */
  _buildTable(snapIdx, pivotRow = -1, pivotCol = -1, opts = {}) {
    const { selRows = false, selCols = false, stepIdx } = opts;
    const solver = this.solver;
    const tab    = solver.getTableau(snapIdx);
    const basis  = solver.getBasis(snapIdx);
    const hdrs   = this._colHeaders();
    const bCol   = solver.bCol;

    const wrap  = document.createElement('div');
    wrap.className = 'simplex-table-wrap';
    const table = document.createElement('table');
    table.className = 'simplex-table';

    // thead
    const thead = document.createElement('thead');
    const hrow  = document.createElement('tr');
    const thBasis = document.createElement('th');
    thBasis.className = 'col-basis';
    thBasis.textContent = 'Базис';
    hrow.appendChild(thBasis);

    hdrs.forEach((h, j) => {
      const th = document.createElement('th');
      th.textContent = h;
      if (j === bCol) th.className = 'col-b';
      if (selCols && j < bCol) {
        th.classList.add('col-selectable');
        th.dataset.col = j;
        th.style.cursor = 'pointer';
      }
      if (j === pivotCol) th.classList.add('pivot-col');
      hrow.appendChild(th);
    });
    thead.appendChild(hrow);
    table.appendChild(thead);

    // tbody
    const tbody = document.createElement('tbody');
    tab.forEach((row, i) => {
      const tr = document.createElement('tr');

      // Objective row label
      if (i === solver.m) {
        tr.classList.add('row-delta');
        const tdL = document.createElement('td');
        tdL.className = 'col-basis';
        tdL.textContent = 'Δ';
        tr.appendChild(tdL);
      } else {
        if (selRows) {
          tr.classList.add('row-selectable');
          tr.dataset.row = i;
        }
        if (i === pivotRow) tr.classList.add('pivot-row');

        const tdL = document.createElement('td');
        tdL.className = 'col-basis';
        tdL.textContent = solver.varName(basis[i]);
        tr.appendChild(tdL);
      }

      row.forEach((val, j) => {
        const td = document.createElement('td');
        td.textContent = DualSimplexSolver.fmt(val);
        if (j === bCol) td.className = 'col-b';
        if (i !== solver.m && selCols && j < bCol) td.classList.add('col-selectable');
        if (j === pivotCol && i !== solver.m) td.classList.add('pivot-col');
        if (i === pivotRow) td.classList.add('pivot-row');
        if (i === pivotRow && j === pivotCol) td.classList.add('pivot-element');
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return { wrap, table, tbody, thead };
  }

  /* ── Table where every cell is a blank input ──
     Used for the canonical-form interactive step.
     Returns { wrap, inputCells } where inputCells = [{inp, correctVal}]
  ─────────────────────────────────────────────────── */
  _buildFullInputTable(snapIdx) {
    const solver = this.solver;
    const tab    = solver.getTableau(snapIdx);
    const basis  = solver.getBasis(snapIdx);
    const hdrs   = this._colHeaders();
    const bCol   = solver.bCol;
    const inputCells = [];

    const wrap  = document.createElement('div');
    wrap.className = 'simplex-table-wrap';
    const table = document.createElement('table');
    table.className = 'simplex-table';

    const thead = document.createElement('thead');
    const hrow  = document.createElement('tr');
    const thB   = document.createElement('th');
    thB.className = 'col-basis'; thB.textContent = 'Базис';
    hrow.appendChild(thB);
    hdrs.forEach((h, j) => {
      const th = document.createElement('th');
      th.textContent = h;
      if (j === bCol) th.className = 'col-b';
      hrow.appendChild(th);
    });
    thead.appendChild(hrow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    tab.forEach((row, i) => {
      const tr = document.createElement('tr');
      const tdLabel = document.createElement('td');
      tdLabel.className = 'col-basis';
      if (i === solver.m) {
        tr.classList.add('row-delta');
        tdLabel.textContent = 'Δ';
      } else {
        tdLabel.textContent = solver.varName(basis[i]);
      }
      tr.appendChild(tdLabel);

      row.forEach((val, j) => {
        const td = document.createElement('td');
        if (j === bCol) td.className = 'col-b';
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'cell-input';
        inp.placeholder = '?';
        td.appendChild(inp);
        inputCells.push({ inp, correctVal: val });
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return { wrap, inputCells };
  }

  /* ── Table where b-column and Δ-row are blank inputs ──
     A-matrix cells are read-only text.
     Returns { wrap, inputCells } where inputCells = [{inp, correctVal}]
  ─────────────────────────────────────────────────── */
  _buildInputTable(snapIdx) {
    const solver = this.solver;
    const tab    = solver.getTableau(snapIdx);
    const basis  = solver.getBasis(snapIdx);
    const hdrs   = this._colHeaders();
    const bCol   = solver.bCol;

    const inputCells = [];

    const wrap  = document.createElement('div');
    wrap.className = 'simplex-table-wrap';
    const table = document.createElement('table');
    table.className = 'simplex-table';

    // thead
    const thead = document.createElement('thead');
    const hrow  = document.createElement('tr');
    const thB   = document.createElement('th');
    thB.className = 'col-basis';
    thB.textContent = 'Базис';
    hrow.appendChild(thB);
    hdrs.forEach((h, j) => {
      const th = document.createElement('th');
      th.textContent = h;
      if (j === bCol) th.className = 'col-b';
      hrow.appendChild(th);
    });
    thead.appendChild(hrow);
    table.appendChild(thead);

    // tbody
    const tbody = document.createElement('tbody');
    tab.forEach((row, i) => {
      const tr = document.createElement('tr');
      const tdLabel = document.createElement('td');
      tdLabel.className = 'col-basis';
      if (i === solver.m) {
        tr.classList.add('row-delta');
        tdLabel.textContent = 'Δ';
      } else {
        tdLabel.textContent = solver.varName(basis[i]);
      }
      tr.appendChild(tdLabel);

      row.forEach((val, j) => {
        const td = document.createElement('td');
        if (j === bCol) td.className = 'col-b';

        const needsInput = (j === bCol) || (i === solver.m);
        if (needsInput) {
          const inp = document.createElement('input');
          inp.type = 'text';
          inp.className = 'cell-input';
          inp.placeholder = '?';
          td.appendChild(inp);
          inputCells.push({ inp, correctVal: val });
        } else {
          td.textContent = DualSimplexSolver.fmt(val);
        }

        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    return { wrap, inputCells };
  }

  /* ── Collapsible help panel (rules + animated demo) ── */
  _buildHelpPanel(stage) {
    const cfg = HelpContent[stage]();

    const panel = document.createElement('div');
    panel.className = 'help-panel';

    const toggle = document.createElement('button');
    toggle.className = 'btn-help-toggle';
    toggle.innerHTML =
      `<span>📖 ${cfg.title} — правила та приклад</span>` +
      `<span class="help-chevron">▼</span>`;

    const content = document.createElement('div');
    content.className = 'help-content hidden';

    toggle.addEventListener('click', () => {
      const nowHidden = content.classList.toggle('hidden');
      toggle.classList.toggle('open', !nowHidden);
    });

    // Rules
    const rulesWrap = document.createElement('div');
    rulesWrap.className = 'help-rules';
    const rTitle = document.createElement('div');
    rTitle.className = 'help-section-title'; rTitle.textContent = 'Правила';
    rulesWrap.appendChild(rTitle);
    cfg.rules.forEach(r => {
      const el = document.createElement('div'); el.className = 'help-rule'; el.innerHTML = r;
      rulesWrap.appendChild(el);
    });
    content.appendChild(rulesWrap);

    // Demo
    const demo = document.createElement('div');
    demo.className = 'help-demo';

    const dTitle = document.createElement('div');
    dTitle.className = 'demo-title'; dTitle.textContent = 'Покроковий приклад';
    demo.appendChild(dTitle);

    if (cfg.problem) {
      const prob = document.createElement('pre'); prob.className = 'demo-problem'; prob.textContent = cfg.problem;
      demo.appendChild(prob);
    }

    demo.appendChild(cfg.table);

    const stepText = document.createElement('div');
    stepText.className = 'demo-step-text'; stepText.textContent = 'Натисніть ▶ для перегляду прикладу';
    demo.appendChild(stepText);

    const formulaEl = document.createElement('div');
    formulaEl.className = 'demo-formula hidden';
    demo.appendChild(formulaEl);

    const controls = document.createElement('div'); controls.className = 'demo-controls';
    const btnPlay = document.createElement('button'); btnPlay.className = 'btn-demo';           btnPlay.textContent = '▶ Програти';
    const btnPrev = document.createElement('button'); btnPrev.className = 'btn-demo secondary'; btnPrev.textContent = '‹ Назад';
    const btnNext = document.createElement('button'); btnNext.className = 'btn-demo secondary'; btnNext.textContent = 'Далі ›';
    const counter = document.createElement('span');   counter.className = 'demo-counter';
    controls.appendChild(btnPlay);
    controls.appendChild(btnPrev);
    controls.appendChild(btnNext);
    controls.appendChild(counter);
    demo.appendChild(controls);
    content.appendChild(demo);

    panel.appendChild(toggle);
    panel.appendChild(content);

    const anim = new StepAnimator(demo);
    anim.load(cfg.steps);
    btnPlay.addEventListener('click', () => anim.start());
    btnPrev.addEventListener('click', () => anim.prev());
    btnNext.addEventListener('click', () => anim.next());

    return panel;
  }

  /* ── Create a step card shell ────────────── */
  _makeCard(stepNum, title) {
    const card = document.createElement('div');
    card.className = 'step-card';
    card.id = `step-card-${stepNum}`;

    const hdr = document.createElement('div');
    hdr.className = 'step-header';

    const badge = document.createElement('span');
    badge.className = 'step-badge';
    badge.textContent = stepNum;

    const ttl = document.createElement('span');
    ttl.className = 'step-title';
    ttl.textContent = title;

    hdr.appendChild(badge);
    hdr.appendChild(ttl);
    card.appendChild(hdr);

    const body = document.createElement('div');
    body.className = 'step-body';
    card.appendChild(body);

    return { card, body };
  }

  /* ── Feedback line helper ─── */
  _feedback(text, type) {
    const el = document.createElement('div');
    el.className = `phase-feedback ${type}`;
    el.textContent = text;
    return el;
  }

  /* ── Render iteration step N ─────────────── */
  _renderStep(stepIdx) {
    const solver   = this.solver;
    const steps    = this.result.steps;
    const isLast   = stepIdx >= steps.length; // no more iterations needed
    const snapIdx  = isLast ? solver.history.length - 1 : steps[stepIdx].snapIdx;
    const cardNum  = stepIdx + 1;
    const isOptimal = solver.isPrimalFeasible(solver.history[snapIdx]);

    const { card, body } = this._makeCard(cardNum, `Ітерація ${cardNum} — перевірка оптимальності`);
    this.output.appendChild(card);

    // ── Phase 1: current table ──
    const tablePhase = document.createElement('div');
    tablePhase.className = 'phase';

    if (this.mode === 'student') {
      // Table was already verified in the previous step (canonical form or Gauss-Jordan recalc) — show read-only
      const tq = document.createElement('p');
      tq.className = 'phase-question';
      tq.textContent = stepIdx === 0
        ? 'Поточна симплекс-таблиця (перевірена на кроці канонічної форми):'
        : 'Поточна симплекс-таблиця (результат попереднього кроку Гаусса-Жордана):';
      tablePhase.appendChild(tq);
      tablePhase.appendChild(this._buildTable(snapIdx).wrap);
      body.appendChild(tablePhase);

      const optPhase = document.createElement('div');
      optPhase.className = 'phase';
      this._buildOptPhase(optPhase, snapIdx, stepIdx, body, card);
      body.appendChild(optPhase);

      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }

    // Guide mode: show full read-only table immediately
    const tq2 = document.createElement('p');
    tq2.className = 'phase-question';
    tq2.textContent = 'Поточна симплекс-таблиця:';
    tablePhase.appendChild(tq2);
    const { wrap: tableWrap } = this._buildTable(snapIdx);
    tablePhase.appendChild(tableWrap);
    body.appendChild(tablePhase);

    // ── Phase 2: optimality question (visible immediately in guide/non-student path) ──
    const optPhase = document.createElement('div');
    optPhase.className = 'phase';
    this._buildOptPhase(optPhase, snapIdx, stepIdx, body, card);
    body.appendChild(optPhase);

    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* ── Shared: populate an opt-check phase element ── */
  _buildOptPhase(optPhase, snapIdx, stepIdx, body, card) {
    const solver = this.solver;

    const optQ = document.createElement('p');
    optQ.className = 'phase-question';
    optQ.textContent = 'Чи є поточний план оптимальним? (перевірте, чи всі bᵢ ≥ 0)';
    optPhase.appendChild(optQ);
    optPhase.appendChild(this._buildHelpPanel('optimality'));

    const btnGroup = document.createElement('div');
    btnGroup.className = 'btn-group';

    const btnYes = document.createElement('button');
    btnYes.className = 'btn btn-choice';
    btnYes.textContent = 'Так, план оптимальний';

    const btnNo = document.createElement('button');
    btnNo.className = 'btn btn-choice';
    btnNo.textContent = 'Ні, потрібна ітерація';

    btnGroup.appendChild(btnYes);
    btnGroup.appendChild(btnNo);
    optPhase.appendChild(btnGroup);

    const optFeedback = document.createElement('div');
    optFeedback.className = 'phase-feedback hidden';
    optPhase.appendChild(optFeedback);

    const onOptChoice = (userSaysOptimal) => {
      const correct = solver.validateOptimalityAnswer(snapIdx, userSaysOptimal);
      [btnYes, btnNo].forEach(b => b.disabled = true);

      if (!correct) {
        const chosen = userSaysOptimal ? btnYes : btnNo;
        chosen.classList.add('selected-wrong');
        optFeedback.className = 'phase-feedback error';
        optFeedback.textContent = userSaysOptimal
          ? 'Невірно. Є від\'ємні bᵢ — план ще не оптимальний.'
          : 'Невірно. Всі bᵢ ≥ 0 — план вже оптимальний.';
        setTimeout(() => {
          [btnYes, btnNo].forEach(b => { b.disabled = false; b.classList.remove('selected-wrong'); });
          optFeedback.className = 'phase-feedback hidden';
        }, 1200);
        return;
      }

      const chosen = userSaysOptimal ? btnYes : btnNo;
      chosen.classList.add('selected');
      optFeedback.className = 'phase-feedback success';
      optFeedback.textContent = userSaysOptimal
        ? 'Правильно! Всі bᵢ ≥ 0 — розв\'язок знайдено.'
        : 'Правильно! Є від\'ємні bᵢ — виконуємо ітерацію.';

      if (userSaysOptimal) {
        setTimeout(() => this._showFinalResult(), 400);
      } else {
        setTimeout(() => this._renderPivotRowPhase(body, stepIdx, snapIdx, card), 500);
      }
    };

    btnYes.addEventListener('click', () => onOptChoice(true));
    btnNo .addEventListener('click', () => onOptChoice(false));
  }

  /* ── Phase 3: pivot row selection ────────── */
  _renderPivotRowPhase(body, stepIdx, snapIdx, card) {
    const solver = this.solver;

    // When stepIdx >= steps.length the solver found no valid pivot column → infeasible.
    // We still need to show the pivot-row selection so the student sees WHY it's infeasible.
    const isTerminalInfeasible = stepIdx >= this.result.steps.length;

    let pivotRow;
    if (!isTerminalInfeasible) {
      pivotRow = this.result.steps[stepIdx].pivotRow;
    } else {
      // Derive most-negative b row directly from the snapshot
      const tab = solver.getTableau(snapIdx);
      pivotRow = -1;
      let mostNeg = -DualSimplexSolver.EPSILON;
      for (let i = 0; i < solver.m; i++) {
        const bi = tab[i][solver.bCol];
        if (bi < mostNeg) { mostNeg = bi; pivotRow = i; }
      }
    }

    const phase = document.createElement('div');
    phase.className = 'phase';

    const q = document.createElement('p');
    q.className = 'phase-question';
    q.textContent = 'Оберіть ведучий рядок (натисніть на рядок із найбільш від\'ємним bᵢ):';
    phase.appendChild(q);
    phase.appendChild(this._buildHelpPanel('pivotRow'));

    const { wrap, tbody } = this._buildTable(snapIdx, -1, -1, { selRows: true });
    phase.appendChild(wrap);

    const fb = document.createElement('div');
    fb.className = 'phase-feedback hidden';
    phase.appendChild(fb);

    body.appendChild(phase);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    let rowMisses = 0;
    let rowSelected = false;
    tbody.querySelectorAll('tr.row-selectable').forEach(tr => {
      tr.addEventListener('click', () => {
        if (rowSelected) return;
        const userRow = parseInt(tr.dataset.row, 10);
        if (userRow === pivotRow) {
          rowSelected = true;
          tbody.querySelectorAll('tr.row-selectable').forEach(r => r.classList.remove('row-selectable'));
          tr.classList.add('pivot-row');

          if (isTerminalInfeasible) {
            // All a_rj ≥ 0 in this row → no pivot column exists → infeasible
            fb.className = 'phase-feedback error';
            fb.textContent =
              `Рядок ${pivotRow + 1} — вірно. Але у цьому рядку всі елементи aᵣⱼ ≥ 0: ` +
              'ведучого стовпця не існує — допустима область порожня (задача нерозв\'язна).';
            setTimeout(() => this._showFinalResult(), 1200);
          } else {
            fb.className = 'phase-feedback success';
            fb.textContent = `Правильно! Ведучий рядок — рядок ${pivotRow + 1} (${solver.varName(solver.getBasis(snapIdx)[pivotRow])}).`;
            setTimeout(() => this._renderPivotColPhase(body, stepIdx, snapIdx, pivotRow, card), 600);
          }
        } else {
          rowMisses++;
          const tab   = solver.getTableau(snapIdx);
          const userB = DualSimplexSolver.fmt(tab[userRow][solver.bCol]);
          let hint = `Невірно. b${userRow + 1} = ${userB}.`;
          if (rowMisses === 1) {
            hint += ' Підказка: перегляньте стовпець b і знайдіть найменше (найбільш від\'ємне) значення.';
          } else {
            const negRows = [];
            for (let i = 0; i < solver.m; i++) {
              if (tab[i][solver.bCol] < -DualSimplexSolver.EPSILON) negRows.push(i + 1);
            }
            hint += ` Від'ємні bᵢ є у рядку(ах): ${negRows.join(', ')}. Оберіть рядок з найменшим значенням.`;
          }
          fb.className = 'phase-feedback error';
          fb.textContent = hint;
        }
      });
    });
  }

  /* ── Phase 4: pivot column selection ─────── */
  _renderPivotColPhase(body, stepIdx, snapIdx, pivotRow, card) {
    const solver   = this.solver;
    const pivotCol = this.result.steps[stepIdx].pivotCol;

    const phase = document.createElement('div');
    phase.className = 'phase';

    const q = document.createElement('p');
    q.className = 'phase-question';
    q.textContent = 'Оберіть ведучий стовпець (натисніть заголовок стовпця; мін. |Δⱼ / aᵣⱼ| серед aᵣⱼ < 0):';
    phase.appendChild(q);
    phase.appendChild(this._buildHelpPanel('pivotCol'));

    const { wrap, thead } = this._buildTable(snapIdx, pivotRow, -1, { selCols: true });
    phase.appendChild(wrap);

    const fb = document.createElement('div');
    fb.className = 'phase-feedback hidden';
    phase.appendChild(fb);

    body.appendChild(phase);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    let colMisses = 0;
    let colSelected = false;
    thead.querySelectorAll('th.col-selectable').forEach(th => {
      th.addEventListener('click', () => {
        if (colSelected) return;
        const userCol = parseInt(th.dataset.col, 10);
        if (userCol === pivotCol) {
          colSelected = true;
          fb.className = 'phase-feedback success';
          fb.textContent = `Правильно! Ведучий стовпець — ${solver.varName(pivotCol)}.`;
          thead.querySelectorAll('th.col-selectable').forEach(h => h.classList.remove('col-selectable'));
          wrap.replaceWith(this._buildTable(snapIdx, pivotRow, pivotCol).wrap);
          setTimeout(() => this._renderRecalcPhase(body, stepIdx, snapIdx, pivotRow, pivotCol, card), 600);
        } else {
          colMisses++;
          const tab = solver.getTableau(snapIdx);
          const arj = tab[pivotRow][userCol];
          let hint;
          if (arj >= -DualSimplexSolver.EPSILON) {
            hint = `Невірно. a${pivotRow+1},${userCol+1} = ${DualSimplexSolver.fmt(arj)} ≥ 0 — цей стовпець не можна обрати (потрібен від'ємний елемент у ведучому рядку).`;
          } else {
            const dj    = tab[solver.m][userCol];
            const ratio = Math.abs(dj / arj);
            hint = `Невірно. Відношення |Δ/a| для ${solver.varName(userCol)} = |${DualSimplexSolver.fmt(dj)}/${DualSimplexSolver.fmt(arj)}| = ${DualSimplexSolver.fmt(ratio)}.`;
            if (colMisses >= 2) {
              hint += ' Підказка: обчисліть таке відношення для кожного від\'ємного aᵣⱼ і оберіть стовпець з найменшим результатом.';
            }
          }
          fb.className = 'phase-feedback error';
          fb.textContent = hint;
        }
      });
    });
  }

  /* ── Phase 5: manual recalculation table ─── */
  _renderRecalcPhase(body, stepIdx, snapIdx, pivotRow, pivotCol, card) {
    const solver   = this.solver;
    const nextSnap = snapIdx + 1;
    const correct  = solver.getTableau(nextSnap);
    const nextBasis = solver.getBasis(nextSnap);
    const hdrs     = this._colHeaders();
    const bCol     = solver.bCol;

    const phase = document.createElement('div');
    phase.className = 'phase';

    const q = document.createElement('p');
    q.className = 'phase-question';
    q.textContent = 'Заповніть нову симплекс-таблицю після виконання кроку Гаусса-Жордана:';
    phase.appendChild(q);
    phase.appendChild(this._buildHelpPanel('gauss'));

    // Build editable table
    const wrap  = document.createElement('div');
    wrap.className = 'simplex-table-wrap';
    const table = document.createElement('table');
    table.className = 'simplex-table editable';

    // thead (read-only headers)
    const thead = document.createElement('thead');
    const hrow  = document.createElement('tr');
    const thB   = document.createElement('th'); thB.className = 'col-basis'; thB.textContent = 'Базис';
    hrow.appendChild(thB);
    hdrs.forEach((h, j) => {
      const th = document.createElement('th');
      th.textContent = h;
      if (j === bCol) th.className = 'col-b';
      hrow.appendChild(th);
    });
    thead.appendChild(hrow);
    table.appendChild(thead);

    // tbody with inputs
    const tbody = document.createElement('tbody');
    const inputs = []; // inputs[i][j]

    correct.forEach((row, i) => {
      const tr = document.createElement('tr');
      const tdLabel = document.createElement('td');
      tdLabel.className = 'col-basis';

      if (i === solver.m) {
        tr.classList.add('row-delta');
        tdLabel.textContent = 'Δ';
      } else {
        tdLabel.textContent = solver.varName(nextBasis[i]);
      }
      tr.appendChild(tdLabel);

      const rowInputs = [];
      row.forEach((val, j) => {
        const td = document.createElement('td');
        if (j === bCol) td.className = 'col-b';

        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'cell-input';
        inp.placeholder = '?';
        inp.dataset.row = i;
        inp.dataset.col = j;
        td.appendChild(inp);
        rowInputs.push(inp);

        tr.appendChild(td);
      });

      inputs.push(rowInputs);
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    phase.appendChild(wrap);

    // Controls
    const controls = document.createElement('div');
    controls.className = 'recalc-controls';

    const btnCheck = document.createElement('button');
    btnCheck.className = 'btn btn-primary';
    btnCheck.textContent = 'Перевірити';

    controls.appendChild(btnCheck);
    phase.appendChild(controls);

    const recalcFb = document.createElement('div');
    recalcFb.className = 'phase-feedback hidden';
    phase.appendChild(recalcFb);

    body.appendChild(phase);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    let checkMisses = 0;

    btnCheck.addEventListener('click', () => {
      const userGrid = inputs.map(rowArr => rowArr.map(inp => inp.value));
      const grid     = solver.validateTableau(nextSnap, userGrid);
      let allCorrect = true;
      let wrongCount = 0;

      inputs.forEach((rowArr, i) => {
        rowArr.forEach((inp, j) => {
          inp.classList.remove('correct', 'wrong');
          if (grid[i][j]) {
            inp.classList.add('correct');
          } else {
            inp.classList.add('wrong');
            allCorrect = false;
            wrongCount++;
          }
        });
      });

      if (allCorrect) {
        recalcFb.className = 'phase-feedback success';
        recalcFb.textContent = 'Чудово! Таблицю заповнено правильно. Переходимо до наступного кроку.';
        btnCheck.disabled = true;
        setTimeout(() => {
          this.stepIdx = stepIdx + 1;
          this._renderStep(this.stepIdx);
        }, 700);
      } else {
        checkMisses++;
        let msg = `${wrongCount} ${wrongCount === 1 ? 'значення невірне' : 'значень невірних'} (позначені червоним).`;
        if (checkMisses === 1) {
          msg += ' Ведучий рядок ÷ ведучий елемент; інші: новий = старий − коеф × ведучий. Скористайтесь панеллю «Правила та приклад» вище.';
        } else {
          msg += ' Перевірте знаки. Коефіцієнт для рядка i = a[i][ведучий стовп] зі СТАРОЇ таблиці.';
        }
        recalcFb.className = 'phase-feedback error';
        recalcFb.textContent = msg;
      }
    });
  }

  /* ── Post-solve explanation card (student mode, optimal only) ── */
  _renderSolutionExplanation() {
    const solver = this.solver;
    const res    = this.result;

    const { card, body } = this._makeCard('★', 'Розбір розв\'язку');
    this.output.appendChild(card);

    // ── Section 1: optimal variable values ──────────────────────
    const s1 = document.createElement('div'); s1.className = 'phase';
    const s1q = document.createElement('p'); s1q.className = 'phase-question';
    s1q.textContent = 'Оптимальний план:';
    s1.appendChild(s1q);

    const chipsWrap = document.createElement('div');
    chipsWrap.className = 'solution-vars';
    Array.from(res.x).forEach((v, j) => {
      const chip = document.createElement('div');
      const isZero = Math.abs(v) < DualSimplexSolver.EPSILON;
      chip.className = 'solution-chip' + (isZero ? ' zero' : '');

      const varSpan = document.createElement('span'); varSpan.className = 'sol-var';
      varSpan.textContent = `x${j + 1}`;
      const eqSpan = document.createElement('span'); eqSpan.className = 'sol-eq';
      eqSpan.textContent = '=';
      const valSpan = document.createElement('span'); valSpan.className = 'sol-val';
      valSpan.textContent = DualSimplexSolver.fmt(v);

      chip.appendChild(varSpan);
      chip.appendChild(eqSpan);
      chip.appendChild(valSpan);
      chipsWrap.appendChild(chip);
    });
    s1.appendChild(chipsWrap);
    body.appendChild(s1);

    // ── Section 2: objective function verification ───────────────
    const s2 = document.createElement('div'); s2.className = 'phase';
    const s2q = document.createElement('p'); s2q.className = 'phase-question';
    s2q.textContent = 'Перевірка цільової функції:';
    s2.appendChild(s2q);

    const n = solver.n;
    const cOrig = solver._cOrig;
    const xVals = res.x;

    // Build the F(x*) = ... line
    const termParts = [];
    const productParts = [];
    for (let j = 0; j < n; j++) {
      termParts.push(`${DualSimplexSolver.fmt(cOrig[j])}·x${j + 1}*`);
      productParts.push(`${DualSimplexSolver.fmt(cOrig[j])}·${DualSimplexSolver.fmt(xVals[j])}`);
    }
    const fVal = DualSimplexSolver.fmt(res.objectiveValue);
    const verifyLine =
      `F(x*) = ${termParts.join(' + ')}\n` +
      `      = ${productParts.join(' + ')}\n` +
      `      = ${fVal}`;

    const verifyBox = document.createElement('div');
    verifyBox.className = 'solution-verify';
    verifyBox.textContent = verifyLine;
    s2.appendChild(verifyBox);
    body.appendChild(s2);

    // ── Section 3: iteration summary table ──────────────────────
    if (res.steps.length > 0) {
      const s3 = document.createElement('div'); s3.className = 'phase';
      const s3q = document.createElement('p'); s3q.className = 'phase-question';
      s3q.textContent = 'Зведена таблиця ітерацій:';
      s3.appendChild(s3q);

      const iterTable = document.createElement('table');
      iterTable.className = 'iter-summary-table';

      const thead = document.createElement('thead');
      const hrow  = document.createElement('tr');
      ['Ітерація', 'Ведучий рядок', 'Ведучий стовпець', 'Вийшла зі складу базису', 'Увійшла до базису'].forEach(h => {
        const th = document.createElement('th'); th.textContent = h; hrow.appendChild(th);
      });
      thead.appendChild(hrow);
      iterTable.appendChild(thead);

      const tbody = document.createElement('tbody');
      res.steps.forEach((step, idx) => {
        const tr = document.createElement('tr');
        const basis = solver.getBasis(step.snapIdx);
        const outgoing = solver.varName(basis[step.pivotRow]);
        const incoming = solver.varName(step.pivotCol);

        const tdNum = document.createElement('td'); tdNum.className = 'iter-num'; tdNum.textContent = idx + 1;
        const tdRow = document.createElement('td'); tdRow.textContent = step.pivotRow + 1;
        const tdCol = document.createElement('td'); tdCol.textContent = step.pivotCol + 1;
        const tdOut = document.createElement('td'); tdOut.className = 'iter-out'; tdOut.textContent = outgoing;
        const tdIn  = document.createElement('td'); tdIn.className  = 'iter-in';  tdIn.textContent  = incoming;

        [tdNum, tdRow, tdCol, tdOut, tdIn].forEach(td => tr.appendChild(td));
        tbody.appendChild(tr);
      });
      iterTable.appendChild(tbody);
      s3.appendChild(iterTable);
      body.appendChild(s3);
    }

    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ── Final result banner ────────────────── */
  _showFinalResult() {
    const res    = this.result;
    const solver = this.solver;
    const banner = this.banner;

    if (res.status === 'optimal') {
      banner.className = 'result-banner optimal';

      document.getElementById('resultIcon' ).textContent = '✓';
      document.getElementById('resultTitle').textContent = 'Оптимальний розв\'язок знайдено';

      const vals = Array.from(res.x).map((v, j) =>
        `x${j + 1} = ${DualSimplexSolver.fmt(v)}`
      ).join(', ');

      const obj = DualSimplexSolver.fmt(res.objectiveValue);
      document.getElementById('resultBody').innerHTML =
        `<div class="result-values">${
          Array.from(res.x).map((v, j) =>
            `<span class="result-chip">x${j+1} = ${DualSimplexSolver.fmt(v)}</span>`
          ).join('')
        }</div>` +
        `<div class="result-objective">F(x) = ${obj}</div>`;
    } else if (res.status === 'infeasible') {
      banner.className = 'result-banner infeasible';
      document.getElementById('resultIcon' ).textContent = '✕';
      document.getElementById('resultTitle').textContent = 'Задача не має розв\'язку';
      document.getElementById('resultBody' ).textContent =
        'У ведучому рядку немає від\'ємних елементів aᵣⱼ — множина допустимих розв\'язків порожня.';
    } else {
      banner.className = 'result-banner infeasible';
      document.getElementById('resultIcon' ).textContent = '!';
      document.getElementById('resultTitle').textContent = 'Перевищено ліміт ітерацій';
      document.getElementById('resultBody' ).textContent =
        'Алгоритм не збігся за відведену кількість ітерацій. Перевірте коректність введених даних.';
    }

    banner.classList.remove('hidden');
    banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    if (this.mode === 'student' && res.status === 'optimal') {
      setTimeout(() => this._renderSolutionExplanation(), 300);
    }
  }
}


/* ═══════════════════════════════════════════════
   Bootstrap
═══════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  new InputManager();
});
