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

  /* ── Construction ───────────────────────────
     c      : Float64Array(n)   objective coefficients
     A      : Float64Array(m*n) constraint matrix, row-major
     b      : Float64Array(m)   RHS
     signs  : string[]          '<=', '>=', '='  per row
     dir    : 'min' | 'max'
  ─────────────────────────────────────────── */
  constructor(c, A, b, signs, dir) {
    this.n    = c.length;
    this.m    = b.length;
    this.dir  = dir;

    // Convert max → min by negating c
    this._cOrig = Float64Array.from(c);
    const cWork = dir === 'max'
      ? Float64Array.from(c, v => -v)
      : Float64Array.from(c);

    // Build initial tableau: (m+1) rows × (n+m+1) cols
    // Rows 0..m-1: constraints; row m: objective row (Δⱼ)
    // Cols 0..n-1: x vars; cols n..n+m-1: slacks; col n+m: b
    this.cols = this.n + this.m + 1; // total columns per row
    this.bCol = this.n + this.m;     // index of RHS column
    this.rows = this.m + 1;          // constraint rows + obj row

    this.tableau = new Float64Array(this.rows * this.cols);

    // Fill constraint rows
    for (let i = 0; i < this.m; i++) {
      const sign = signs[i];
      let rowMult = 1;

      // For >= constraints: multiply row by -1 so b becomes ≤0
      // giving a negative-b starting tableau suited for dual simplex
      if (sign === '>=') rowMult = -1;

      for (let j = 0; j < this.n; j++) {
        this._set(i, j, rowMult * A[i * this.n + j]);
      }
      // slack variable for this row
      this._set(i, this.n + i, rowMult * (sign === '>=' ? -1 : 1));
      // RHS
      this._set(i, this.bCol, rowMult * b[i]);
    }

    // Objective row: reduced costs = cⱼ (slacks have cost 0)
    for (let j = 0; j < this.n; j++) {
      this._set(this.m, j, cWork[j]);
    }

    // Basis: initially slack variables s₁..sₘ
    this.basis = Array.from({ length: this.m }, (_, i) => this.n + i);

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

  /* Return snapshot k as a plain 2-D array (rows × cols) */
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

  /* Dual feasibility: all Δⱼ ≥ 0 (min problem, after converting max) */
  isDualFeasible(snap) {
    const t = snap ? snap.tableau : this.tableau;
    for (let j = 0; j < this.bCol; j++) {
      if (t[this._idx(this.m, j)] < -DualSimplexSolver.EPSILON) return false;
    }
    return true;
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
    const pivotVal = this._get(pivotRow, pivotCol);
    if (Math.abs(pivotVal) < DualSimplexSolver.EPSILON) {
      throw new Error('Pivot element is effectively zero');
    }

    // Scale pivot row
    for (let j = 0; j < this.cols; j++) {
      this._set(pivotRow, j, this._get(pivotRow, j) / pivotVal);
    }

    // Eliminate pivot column from all other rows
    for (let i = 0; i < this.rows; i++) {
      if (i === pivotRow) continue;
      const factor = this._get(i, pivotCol);
      if (Math.abs(factor) < DualSimplexSolver.EPSILON) continue;
      for (let j = 0; j < this.cols; j++) {
        this._set(i, j, this._get(i, j) - factor * this._get(pivotRow, j));
      }
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

    this.status = this.isPrimalFeasible(null) ? 'optimal' : 'infeasible';
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
        const uv = parseFloat(userGrid[i][j]);
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
   InputManager — problem form & dimension controls
═══════════════════════════════════════════════ */
class InputManager {
  constructor() {
    this.numVars = 3;
    this.numCons = 3;

    this._bindSteppers();
    this._renderForm();
    this._bindFormButtons();
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

    // Check that dual simplex start condition holds (all Δj >= 0)
    if (!solver.isDualFeasible(solver.history[0])) {
      this._showError(
        'Початкова таблиця не є подвійно-допустимою (є від\'ємні Δⱼ). ' +
        'Двоїстий симплекс-метод застосовний лише якщо всі Δⱼ ≥ 0 в початковій таблиці.'
      );
      return;
    }

    const result = solver.solve();
    const ui = new UIManager(solver, result);
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
   UIManager — step-by-step interactive flow
═══════════════════════════════════════════════ */
class UIManager {
  constructor(solver, result) {
    this.solver  = solver;
    this.result  = result;
    this.stepIdx = 0; // current iteration index (into result.steps)

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
    this._renderStep(0);
  }

  /* ── Column header labels ─── */
  _colHeaders() {
    const { n, m } = this.solver;
    const hdrs = [];
    for (let j = 0; j < n; j++) hdrs.push(`x${j + 1}`);
    for (let k = 0; k < m; k++) hdrs.push(`s${k + 1}`);
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

    // ── Phase 1: show current table (read-only) ──
    const tablePhase = document.createElement('div');
    tablePhase.className = 'phase';
    const tq = document.createElement('p');
    tq.className = 'phase-question';
    tq.textContent = 'Поточна симплекс-таблиця:';
    tablePhase.appendChild(tq);
    const { wrap: tableWrap } = this._buildTable(snapIdx);
    tablePhase.appendChild(tableWrap);
    body.appendChild(tablePhase);

    // ── Phase 2: optimality question ──
    const optPhase = document.createElement('div');
    optPhase.className = 'phase';

    const optQ = document.createElement('p');
    optQ.className = 'phase-question';
    optQ.textContent = 'Чи є поточний план оптимальним? (перевірте, чи всі bᵢ ≥ 0)';
    optPhase.appendChild(optQ);

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

    body.appendChild(optPhase);

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
        // Re-enable after short delay
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

    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* ── Phase 3: pivot row selection ────────── */
  _renderPivotRowPhase(body, stepIdx, snapIdx, card) {
    const solver    = this.solver;
    const pivotRow  = this.result.steps[stepIdx].pivotRow;

    const phase = document.createElement('div');
    phase.className = 'phase';

    const q = document.createElement('p');
    q.className = 'phase-question';
    q.textContent = 'Оберіть ведучий рядок (натисніть на рядок із найбільш від\'ємним bᵢ):';
    phase.appendChild(q);

    const { wrap, tbody } = this._buildTable(snapIdx, -1, -1, { selRows: true });
    phase.appendChild(wrap);

    const fb = document.createElement('div');
    fb.className = 'phase-feedback hidden';
    phase.appendChild(fb);

    body.appendChild(phase);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Attach click handlers to data rows (not delta row)
    tbody.querySelectorAll('tr.row-selectable').forEach(tr => {
      tr.addEventListener('click', () => {
        const userRow = parseInt(tr.dataset.row, 10);
        if (userRow === pivotRow) {
          fb.className = 'phase-feedback success';
          fb.textContent = `Правильно! Ведучий рядок — рядок ${pivotRow + 1} (${solver.varName(solver.getBasis(snapIdx)[pivotRow])}).`;
          tbody.querySelectorAll('tr.row-selectable').forEach(r => r.classList.remove('row-selectable'));
          tr.classList.add('pivot-row');
          setTimeout(() => this._renderPivotColPhase(body, stepIdx, snapIdx, pivotRow, card), 600);
        } else {
          fb.className = 'phase-feedback error';
          fb.textContent = `Невірно. Рядок ${userRow + 1} не має найбільш від'ємного bᵢ. Спробуйте ще.`;
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

    const { wrap, thead } = this._buildTable(snapIdx, pivotRow, -1, { selCols: true });
    phase.appendChild(wrap);

    const fb = document.createElement('div');
    fb.className = 'phase-feedback hidden';
    phase.appendChild(fb);

    body.appendChild(phase);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    thead.querySelectorAll('th.col-selectable').forEach(th => {
      th.addEventListener('click', () => {
        const userCol = parseInt(th.dataset.col, 10);
        if (userCol === pivotCol) {
          fb.className = 'phase-feedback success';
          fb.textContent = `Правильно! Ведучий стовпець — ${solver.varName(pivotCol)}.`;
          thead.querySelectorAll('th.col-selectable').forEach(h => h.classList.remove('col-selectable'));
          // Re-render table with full pivot highlight
          wrap.replaceWith(this._buildTable(snapIdx, pivotRow, pivotCol).wrap);
          setTimeout(() => this._renderRecalcPhase(body, stepIdx, snapIdx, pivotRow, pivotCol, card), 600);
        } else {
          fb.className = 'phase-feedback error';
          fb.textContent = `Невірно. Стовпець ${solver.varName(userCol)} не є ведучим. Спробуйте ще.`;
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

        // Pivot row is given (read-only) to match textbook convention
        if (i === pivotRow) {
          const inp = document.createElement('input');
          inp.type = 'text';
          inp.className = 'cell-input';
          inp.value = DualSimplexSolver.fmt(val);
          inp.readOnly = true;
          td.appendChild(inp);
          rowInputs.push(inp);
        } else {
          const inp = document.createElement('input');
          inp.type = 'text';
          inp.className = 'cell-input';
          inp.placeholder = '?';
          inp.dataset.row = i;
          inp.dataset.col = j;
          td.appendChild(inp);
          rowInputs.push(inp);
        }

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

    const btnHint = document.createElement('button');
    btnHint.className = 'btn btn-hint';
    btnHint.textContent = 'Підказка / Показати відповідь';

    controls.appendChild(btnCheck);
    controls.appendChild(btnHint);
    phase.appendChild(controls);

    const recalcFb = document.createElement('div');
    recalcFb.className = 'phase-feedback hidden';
    phase.appendChild(recalcFb);

    body.appendChild(phase);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // ── Check button ──
    btnCheck.addEventListener('click', () => {
      const userGrid = inputs.map(rowArr => rowArr.map(inp => inp.value));
      const grid     = solver.validateTableau(nextSnap, userGrid);
      let allCorrect = true;

      inputs.forEach((rowArr, i) => {
        rowArr.forEach((inp, j) => {
          if (inp.readOnly) return;
          inp.classList.remove('correct', 'wrong');
          if (grid[i][j]) {
            inp.classList.add('correct');
          } else {
            inp.classList.add('wrong');
            allCorrect = false;
          }
        });
      });

      if (allCorrect) {
        recalcFb.className = 'phase-feedback success';
        recalcFb.textContent = 'Чудово! Таблицю заповнено правильно. Переходимо до наступного кроку.';
        btnCheck.disabled = true;
        btnHint.disabled  = true;
        setTimeout(() => {
          this.stepIdx = stepIdx + 1;
          this._renderStep(this.stepIdx);
        }, 700);
      } else {
        recalcFb.className = 'phase-feedback error';
        recalcFb.textContent = 'Деякі значення невірні (позначені червоним). Перевірте обчислення.';
      }
    });

    // ── Hint button ──
    btnHint.addEventListener('click', () => {
      inputs.forEach((rowArr, i) => {
        rowArr.forEach((inp, j) => {
          if (inp.readOnly) return;
          inp.classList.remove('correct', 'wrong');
          inp.classList.add('revealed');
          inp.value = DualSimplexSolver.fmt(correct[i][j]);
        });
      });
      recalcFb.className = 'phase-feedback info';
      recalcFb.textContent = 'Правильні значення показано. Можете продовжити далі.';
      btnCheck.disabled = false;
    });
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
    } else {
      banner.className = 'result-banner infeasible';
      document.getElementById('resultIcon' ).textContent = '✕';
      document.getElementById('resultTitle').textContent = 'Задача не має розв\'язку';
      document.getElementById('resultBody' ).textContent =
        'Серед від\'ємних aᵣⱼ у ведучому рядку немає від\'ємних елементів — множина допустимих розв\'язків порожня.';
    }

    banner.classList.remove('hidden');
    banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}


/* ═══════════════════════════════════════════════
   Bootstrap
═══════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  new InputManager();
});
