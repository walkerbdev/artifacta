Development
===========

This guide covers development workflows for contributing to Artifacta.

Contributing
------------

Thank you for your interest in contributing to Artifacta! We welcome bug reports, feature requests, and pull requests.

**How to Contribute:**

1. **Report bugs or request features** - Open a `GitHub issue <https://github.com/walkerbdev/artifacta/issues>`_
2. **Fix bugs or add features** - Fork the repo, make changes, and submit a pull request
3. **Improve documentation** - Help make our docs clearer and more helpful

**Pull Request Guidelines:**

- Write clear commit messages describing what and why
- Add tests for new features
- Update documentation if needed
- Run ``pre-commit run --all-files`` before submitting
- Keep PRs focused on a single change

**Code Style:**

We use automated tools to maintain code quality:

- **Python**: ruff, mypy, pydocstyle
- **JavaScript**: eslint
- Pre-commit hooks enforce these automatically

**License:**

By contributing, you agree that your contributions will be licensed under the `Elastic License 2.0 <https://github.com/walkerbdev/artifacta/blob/main/LICENSE>`_.

For more details, see `CONTRIBUTING.md <https://github.com/walkerbdev/artifacta/blob/main/CONTRIBUTING.md>`_.

Database Schema
---------------

Artifacta uses SQLAlchemy with SQLite (default) or PostgreSQL.

**runs** - Experiment runs

.. code-block:: text

   run_id             String (PK)      - Unique run identifier
   name               String (unique)  - Display name (e.g., "run_a3f4b2c1")
   project            String           - Project grouping (optional)
   config_artifact_id String (FK)      - Link to config artifact
   created_at         Integer          - Unix timestamp (ms)

**tags** - Key-value metadata for runs

.. code-block:: text

   id      Integer (PK)  - Auto-increment
   run_id  String (FK)   - References runs.run_id
   key     String        - Tag key (e.g., "git.commit", "user")
   value   String        - Tag value (e.g., "abc123", "brandon")

**structured_data** - Logged primitives (Series, Distribution, etc.)

.. code-block:: text

   id              Integer (PK)  - Auto-increment
   run_id          String (FK)   - References runs.run_id
   name            String        - Primitive name
   primitive_type  String        - Type (Series, Distribution, Scatter, etc.)
   section         String        - Organization section (optional)
   data            Text (JSON)   - Primitive data
   meta            Text (JSON)   - Metadata (optional)
   timestamp       Integer       - Unix timestamp (ms)

**artifacts** - File metadata and content

.. code-block:: text

   artifact_id   String (PK)   - Unique artifact ID (e.g., "art_abc123")
   run_id        String (FK)   - References runs.run_id
   name          String        - Artifact name
   hash          String        - Content hash (SHA256)
   storage_path  String        - File path
   size_bytes    Integer       - File size
   meta          Text (JSON)   - Metadata (optional)
   content       Text (JSON)   - File collection structure (optional)
   created_at    Integer       - Unix timestamp (ms)

**artifact_links** - Links artifacts to runs with role

.. code-block:: text

   link_id      String (PK)  - Unique link ID
   artifact_id  String (FK)  - References artifacts.artifact_id
   run_id       String (FK)  - References runs.run_id
   role         String       - "input" or "output"
   created_at   Integer      - Unix timestamp (ms)

**projects** - Project grouping

.. code-block:: text

   project_id  String (PK)  - Project identifier
   created_at  Integer      - Unix timestamp (ms)
   updated_at  Integer      - Unix timestamp (ms)

**project_notes** - Lab notebook entries

.. code-block:: text

   id          Integer (PK)  - Auto-increment
   project_id  String (FK)   - References projects.project_id
   title       String        - Note title
   content     Text          - Markdown content
   created_at  Integer       - Unix timestamp (ms)
   updated_at  Integer       - Unix timestamp (ms)

**project_note_attachments** - Files attached to notes

.. code-block:: text

   id            Integer (PK)  - Auto-increment
   note_id       Integer (FK)  - References project_notes.id
   real_name     String        - Original filename
   storage_path  String        - Hash-based storage path
   mime_type     String        - MIME type
   filesize      Integer       - File size in bytes
   hash          String        - SHA256 hash
   created_at    Integer       - Unix timestamp (ms)

**Relationships:**

.. code-block:: text

   Run 1:N Tag
   Run 1:N StructuredData
   Run 1:N ArtifactLink N:1 Artifact
   Run 1:1 Artifact (config_artifact_id)
   Project 1:N ProjectNote 1:N ProjectNoteAttachment

**Database Configuration:**

Default: ``sqlite:///./data/runs.db``

For PostgreSQL:

.. code-block:: bash

   # Linux/macOS
   export DATABASE_URI="postgresql://user:pass@host:port/dbname"  # pragma: allowlist secret

   # Windows (PowerShell)
   $env:DATABASE_URI="postgresql://user:pass@host:port/dbname"  # pragma: allowlist secret

   # Windows (cmd)
   set DATABASE_URI=postgresql://user:pass@host:port/dbname

See ``tracking-server/database.py`` for full SQLAlchemy model definitions.

Setting Up Development Environment
-----------------------------------

**Prerequisites:** Python 3.9+, Node.js 16+

**1. Clone the repository:**

.. code-block:: bash

   git clone https://github.com/walkerbdev/artifacta.git
   cd artifacta

**2. Create and activate a virtual environment:**

.. code-block:: bash

   # Create venv
   python3 -m venv venv

   # Activate - Linux/macOS
   source venv/bin/activate

   # Activate - Windows (PowerShell)
   venv\Scripts\Activate.ps1

   # Activate - Windows (cmd)
   venv\Scripts\activate.bat

**3. Install Python dependencies:**

.. code-block:: bash

   pip install -e '.[dev]'

This installs Artifacta and all optional dependencies including PyTorch, TensorFlow, and scientific computing libraries from the ``pyproject.toml`` file.

For generating real test videos (optional, requires FFmpeg):

.. code-block:: bash

   pip install -e '.[dev,video]'

Note: Video artifact logging works without this - test helpers will use placeholder MP4 files instead of generating real videos.

**4. Install and build UI:**

.. code-block:: bash

   npm install
   npm run build

**5. Install pre-commit hooks:**

.. code-block:: bash

   pre-commit install

This sets up git hooks that automatically run code quality checks (linting, formatting, spell checking, secret detection) before each commit.

Code Quality and Pre-commit Hooks
----------------------------------

Artifacta uses pre-commit hooks to maintain code quality. Hooks run automatically on ``git commit``.

**Manually run all hooks:**

.. code-block:: bash

   # Linux/macOS
   source venv/bin/activate
   pre-commit run --all-files

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   pre-commit run --all-files

   # Windows (cmd)
   venv\Scripts\activate.bat
   pre-commit run --all-files

**Pre-commit hooks include:**

- **Ruff** - Fast Python linter and formatter
- **Mypy** - Static type checking for tracking-server
- **Pydocstyle** - Docstring style checker (Google style)
- **ESLint** - JavaScript/React linter with JSDoc enforcement
- **Knip** - Find unused JavaScript exports and dependencies
- **Depcheck** - Find unused npm dependencies
- **Vulture** - Find dead Python code
- **Codespell** - Catch typos in code and documentation
- **Detect-secrets** - Prevent committing API keys and passwords
- **General hooks** - Trailing whitespace, file endings, YAML/JSON/TOML validation

Running Tests
-------------

Artifacta includes two types of tests:

1. **Pytest (Python)** - Unit and integration tests for Python API, autolog, and primitives
2. **Playwright (E2E)** - End-to-end browser tests for the UI

Pytest - Python Tests
~~~~~~~~~~~~~~~~~~~~~~

Artifacta's pytest suite tests Python functionality including autolog integrations, data primitives, and domain-specific features.

**Test Categories:**

- ``tests/autolog/`` - PyTorch, TensorFlow, PyTorch Lightning autolog
- ``tests/domains/`` - Domain-specific primitives (genomics, finance, robotics, computer vision, climate, audio/video, etc.)

**1. Start the server in one terminal:**

.. code-block:: bash

   # Linux/macOS
   source venv/bin/activate
   artifacta ui

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   artifacta ui

   # Windows (cmd)
   venv\Scripts\activate.bat
   artifacta ui

**2. Run tests in another terminal:**

.. code-block:: bash

   # Linux/macOS
   source venv/bin/activate
   pytest tests/

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   pytest tests/

   # Windows (cmd)
   venv\Scripts\activate.bat
   pytest tests/

**Run specific test categories:**

.. code-block:: bash

   pytest tests/autolog/ -v                         # All autolog tests
   pytest tests/autolog/test_pytorch_lightning.py -v # PyTorch Lightning tests
   pytest tests/domains/ -v                          # All domain tests
   pytest tests/domains/test_genomics.py -v          # Genomics primitives

**Run specific test by name:**

.. code-block:: bash

   pytest tests/autolog/ -k "test_checkpoint" -v    # Tests matching "checkpoint"
   pytest tests/domains/ -k "test_roc_curve" -v     # Tests matching "roc_curve"

**Custom server configuration:**

Tests automatically use ``localhost:8000`` on Linux/macOS and ``127.0.0.1:8000`` on Windows. To use a different host/port:

.. code-block:: bash

   # Linux/macOS
   export TRACKING_SERVER_HOST=localhost
   export TRACKING_SERVER_PORT=9000
   pytest tests/

   # Windows (PowerShell) - use 127.0.0.1 instead of localhost
   $env:TRACKING_SERVER_HOST="127.0.0.1"
   $env:TRACKING_SERVER_PORT="9000"
   pytest tests/

   # Windows (cmd) - use 127.0.0.1 instead of localhost
   set TRACKING_SERVER_HOST=127.0.0.1
   set TRACKING_SERVER_PORT=9000
   pytest tests/

Playwright - E2E UI Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~

Playwright tests verify the UI works end-to-end by automating browser interactions. These tests run in a real Chromium browser and test core functionality like run selection, data visualization, and navigation.

**Test Files:**

- ``tests/e2e/core.spec.js`` - Core UI functionality (homepage, navigation, API, sidebar)
- ``tests/e2e/visualization.spec.js`` - Data visualization (plots, tables, artifacts, chat)

**Setup:**

Playwright tests handle server startup/shutdown automatically. No manual server setup required.

**Run all E2E tests:**

.. code-block:: bash

   npm run test:e2e

**Run specific test file:**

.. code-block:: bash

   npm run test:e2e -- tests/e2e/core.spec.js
   npm run test:e2e -- tests/e2e/visualization.spec.js

**Run tests matching pattern:**

.. code-block:: bash

   npm run test:e2e -- --grep "plots tab"      # Tests with "plots tab" in name
   npm run test:e2e -- --grep "API"            # Tests with "API" in name

**Interactive UI mode (debugging):**

.. code-block:: bash

   npm run test:e2e:ui

This opens Playwright's UI where you can step through tests, see screenshots, and debug failures.

**Custom server URL:**

By default, tests use ``http://localhost:8000`` (``http://127.0.0.1:8000`` on Windows). To test against a different URL:

.. code-block:: bash

   # Linux/macOS
   ARTIFACTA_URL=http://localhost:8001 npm run test:e2e

   # Windows (PowerShell) - use 127.0.0.1 instead of localhost
   $env:ARTIFACTA_URL="http://127.0.0.1:8001"; npm run test:e2e

   # Windows (cmd) - use 127.0.0.1 instead of localhost
   set ARTIFACTA_URL=http://127.0.0.1:8001 && npm run test:e2e

**What the tests do:**

1. **Global Setup** (``tests/e2e/setup.js``):
   - Cleans database for fresh state
   - Starts Artifacta server on port 8000
   - Runs example script to populate test data
   - Waits for server health check

2. **Core Tests** (6 tests):
   - Homepage loads successfully
   - Run list displays correctly
   - Navigation between tabs works
   - Health check endpoint returns healthy
   - API returns run data
   - Sidebar is interactive

3. **Visualization Tests** (4 tests):
   - Plots tab renders charts
   - Tables tab shows structured data
   - Artifacts tab displays file list
   - Chat tab loads successfully

4. **Global Teardown** (``tests/e2e/teardown.js``):
   - Stops the server
   - Cleans up background processes

**Test output:**

- ``test-results/`` - Screenshots and traces from failed tests
- ``playwright-report/`` - HTML report with test results

These directories are gitignored and safe to delete.

Building Documentation
----------------------

Artifacta uses Sphinx for Python API documentation and JSDoc for UI component documentation.

**Build the docs:**

.. code-block:: bash

   # Linux/macOS
   source venv/bin/activate
   cd docs
   make html

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   cd docs
   .\make.bat html

   # Windows (cmd)
   venv\Scripts\activate.bat
   cd docs
   make.bat html

This automatically:

1. Generates JSDoc documentation from UI components (``npm run docs:ui``)
2. Builds Sphinx documentation (Python API, user guide, examples)
3. Links both together in ``_build/html/``

**View the docs:**

Open ``docs/_build/html/index.html`` in your browser, or serve them locally:

.. code-block:: bash

   python -m http.server 8080 --directory docs/_build/html

Then navigate to http://localhost:8080.

**Build only Python docs** (skip JSDoc):

.. code-block:: bash

   sphinx-build -M html . _build

**Build only UI docs:**

.. code-block:: bash

   npm run docs:ui

Version Management
------------------

Artifacta uses ``bump-my-version`` to manage version numbers across the codebase.

**Show current version:**

.. code-block:: bash

   # Linux/macOS
   source venv/bin/activate
   bump-my-version show current_version

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   bump-my-version show current_version

   # Windows (cmd)
   venv\Scripts\activate.bat
   bump-my-version show current_version

**Bump version:**

.. code-block:: bash

   # Bump patch version (0.1.0 → 0.1.1)
   bump-my-version bump patch

   # Bump minor version (0.1.0 → 0.2.0)
   bump-my-version bump minor

   # Bump major version (0.1.0 → 1.0.0)
   bump-my-version bump major

**Preview changes (dry run):**

.. code-block:: bash

   bump-my-version bump patch --dry-run

When you bump the version, it automatically:

- Updates version in ``pyproject.toml``
- Updates version in ``docs/conf.py`` (appears in documentation header)
- Updates version in ``tracking-server/cli.py`` (CLI ``--version`` flag)
- Creates a git commit with message "Bump version: X → Y"
- Creates a git tag ``vX.Y.Z``

After bumping, rebuild documentation to reflect the new version in docs.

Publishing to PyPI
-------------------

**Prerequisites:**

1. Get your PyPI API tokens:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

2. Set environment variables:

.. code-block:: bash

   # Linux/macOS
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE           # For PyPI
   export TWINE_TEST_PASSWORD=pypi-YOUR-TEST-TOKEN-HERE # For TestPyPI

   # Windows (PowerShell)
   $env:TWINE_USERNAME="__token__"
   $env:TWINE_PASSWORD="pypi-YOUR-TOKEN-HERE"           # For PyPI
   $env:TWINE_TEST_PASSWORD="pypi-YOUR-TEST-TOKEN-HERE" # For TestPyPI

   # Windows (cmd)
   set TWINE_USERNAME=__token__
   set TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE
   set TWINE_TEST_PASSWORD=pypi-YOUR-TEST-TOKEN-HERE

**Build the distribution:**

.. code-block:: bash

   # Linux/macOS
   source venv/bin/activate
   rm -rf dist/ build/ *.egg-info artifacta.egg-info
   npm install
   npm run build
   python -m build

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   Remove-Item -Recurse -Force dist, build, *.egg-info, artifacta.egg-info -ErrorAction SilentlyContinue
   npm install
   npm run build
   python -m build

   # Windows (cmd)
   venv\Scripts\activate.bat
   if exist dist rmdir /s /q dist
   if exist build rmdir /s /q build
   npm install
   npm run build
   python -m build

This creates ``dist/artifacta-X.Y.Z.tar.gz`` and ``dist/artifacta-X.Y.Z-py3-none-any.whl``.

**Publish to TestPyPI (for testing):**

.. code-block:: bash

   # Upload to TestPyPI
   python -m twine upload --repository testpypi dist/*

   # Test installation from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ artifacta

**Publish to PyPI (production):**

.. code-block:: bash

   # Upload to PyPI
   python -m twine upload dist/*

**Complete release workflow:**

.. code-block:: bash

   # 1. Bump version
   bump-my-version bump patch  # or minor/major

   # 2. Build package
   rm -rf dist/ build/ *.egg-info
   npm run build
   python -m build

   # 3. Test on TestPyPI first
   python -m twine upload --repository testpypi dist/*

   # 4. If test passes, publish to PyPI
   python -m twine upload dist/*

   # 5. Push version tag to GitHub
   git push origin main --tags

**Notes:**

- Always test on TestPyPI before publishing to PyPI
- The UI must be built (``npm run build``) before building the Python package
- Environment variables keep your tokens secure (never commit tokens to git)
- The build includes the pre-built UI from ``artifacta_ui/dist/`` and ``dist/``

UI Static File Serving Architecture
------------------------------------

Artifacta bundles the pre-built React UI into the Python package, enabling single-command installation via ``pip install artifacta`` without requiring Node.js.

**How It Works:**

**1. Build Process**

When you run ``npm run build``, Vite compiles the React application into static assets:

.. code-block:: text

   artifacta_ui/
   ├── __init__.py           # Exports UI_DIST_PATH
   ├── dist/
   │   ├── index.html        # Entry point
   │   └── assets/
   │       ├── *.js          # Bundled JavaScript
   │       └── *.css         # Bundled CSS
   └── index.html            # (legacy, may be removed)

**2. Package Inclusion**

The ``pyproject.toml`` declares ``artifacta_ui`` as a Python package and includes UI assets:

.. code-block:: toml

   [tool.setuptools]
   packages = ["artifacta", "tracking_server", "artifacta_ui"]

   [tool.setuptools.package-data]
   artifacta_ui = ["dist/**/*", "index.html"]

When you run ``python -m build``, setuptools includes these files in the wheel/tarball.

**3. Runtime Path Resolution**

The ``artifacta_ui/__init__.py`` module exports the UI location:

.. code-block:: python

   from pathlib import Path
   UI_DIST_PATH = Path(__file__).parent / 'dist'

When installed via pip, ``__file__`` points to ``site-packages/artifacta_ui/__init__.py``, so ``UI_DIST_PATH`` resolves to the bundled static files inside the Python installation.

**4. Development vs Production Detection**

The ``tracking-server/config.py`` handles both scenarios:

.. code-block:: python

   try:
       from artifacta_ui import UI_DIST_PATH  # pip install
   except ImportError:
       UI_DIST_PATH = PROJECT_ROOT / "dist"  # development

- **Production (pip install):** Imports ``UI_DIST_PATH`` from the installed package
- **Development:** Falls back to local ``dist/`` folder in the repository

**5. FastAPI Static File Serving**

The ``tracking-server/main.py`` serves the UI using FastAPI's ``StaticFiles``:

.. code-block:: python

   from fastapi.staticfiles import StaticFiles
   from fastapi.responses import FileResponse

   if UI_DIST_PATH.exists():
       # Serve bundled assets (JS, CSS, images)
       app.mount("/assets", StaticFiles(directory=UI_DIST_PATH / "assets"))

       # SPA routing: serve index.html for non-API routes
       @app.get("/{full_path:path}")
       async def serve_ui(full_path: str):
           if full_path.startswith("api/") or full_path.startswith("ws/"):
               return None  # Let API routes handle themselves
           return FileResponse(UI_DIST_PATH / "index.html")

- ``/assets/*`` routes serve static JS/CSS files
- All other routes (except ``/api/*`` and ``/ws/*``) serve ``index.html`` for React Router
- No separate web server needed - FastAPI handles everything

**Key Benefits:**

- **Single installation:** ``pip install artifacta`` includes both backend and frontend
- **No build required for users:** UI is pre-built and bundled
- **Development flexibility:** Same code works for local development and pip installs
- **Self-contained:** No external web server or CDN dependencies

**For Maintainers:**

- Always run ``npm run build`` before ``python -m build`` to update bundled UI
- The UI is **not** rebuilt during ``pip install`` - users get the pre-built version
- Changes to React code require rebuilding and republishing to PyPI
