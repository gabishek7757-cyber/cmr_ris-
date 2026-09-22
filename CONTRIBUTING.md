# Contributing to CMR-RIS

Thank you for your interest in contributing to **CMR-RIS (CardioMetabolic-Renal Risk Intelligence System)**! We welcome contributions from data scientists, machine learning engineers, clinical researchers, and software engineers.

---

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/cmr_ris.git
   cd cmr_ris
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the automated test suite:**
   ```bash
   pytest -v --tb=short
   ```

5. **Launch the interactive dashboard locally:**
   ```bash
   streamlit run app.py
   ```

---

## Contribution Guidelines

### 1. Code Style & Quality
- Adhere to **PEP 8** standards.
- Include explicit type hints on function signatures (`def calculate_cmri(p_d: float, p_h: float, p_c: float) -> dict:`).
- Keep docstrings updated with clinical rationale and mathematical context.

### 2. Clinical & Data Integrity
- Never introduce data leakage between train, validation, and calibration splits.
- Any new derived clinical biomarker must cite relevant clinical literature or consensus guidelines.
- Conformal prediction coverage must remain calibrated at $\ge 1 - \alpha$.

### 3. Pull Request Protocol
- Create a feature branch (`git checkout -b feature/your-feature-name`).
- Add comprehensive `pytest` test cases in `tests/` for any new logic.
- Ensure all CI tests pass.
- Open a Pull Request with a clear description of changes and test outcomes.

---

## License
By contributing to CMR-RIS, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
