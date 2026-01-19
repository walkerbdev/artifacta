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

   export DATABASE_URI="postgresql://user:pass@host:port/dbname"  # pragma: allowlist secret

See ``tracking-server/database.py`` for full SQLAlchemy model definitions.

Setting Up Development Environment
-----------------------------------

**Prerequisites:** Python 3.9+, Node.js 16+

**1. Clone the repository:**

.. code-block:: bash

   git clone https://github.com/walkerbdev/artifacta.git
   cd artifacta

**2. Create a virtual environment:**

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate

**3. Install Python dependencies:**

.. code-block:: bash

   pip install -e '.[dev]'

This installs Artifacta and all optional dependencies including PyTorch, TensorFlow, and scientific computing libraries from the ``pyproject.toml`` file.

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

   source venv/bin/activate
   pre-commit run --all-files

**Pre-commit hooks include:**

- **Ruff** - Fast Python linter and formatter
- **Mypy** - Static type checking
- **Pydocstyle** - Docstring style checker (Google style)
- **ESLint** - JavaScript/React linter
- **Knip** - Find unused JavaScript exports and dependencies
- **Vulture** - Find dead Python code
- **Codespell** - Catch typos in code and documentation
- **Detect-secrets** - Prevent committing API keys and passwords
- **General hooks** - Trailing whitespace, file endings, YAML/JSON/TOML validation

Running Tests
-------------

Artifacta includes a comprehensive test suite. Tests require a running Artifacta server.

**1. Start the server in one terminal:**

.. code-block:: bash

   source venv/bin/activate
   artifacta ui

**2. Run tests in another terminal:**

.. code-block:: bash

   source venv/bin/activate
   pytest tests/

**Run specific tests:**

.. code-block:: bash

   pytest tests/autolog/ -v                    # Run autolog tests
   pytest tests/domains/test_primitives.py -v  # Run primitive tests

Tests automatically use ``localhost:8000`` by default. If you're running the server on a different host/port, set environment variables:

.. code-block:: bash

   export TRACKING_SERVER_HOST=0.0.0.0
   export TRACKING_SERVER_PORT=9000
   pytest tests/

Building Documentation
----------------------

Artifacta uses Sphinx for documentation.

**Build the docs:**

.. code-block:: bash

   source venv/bin/activate
   venv/bin/sphinx-build -b html docs docs/_build/html

**View the docs:**

Open ``docs/_build/html/index.html`` in your browser, or serve them locally:

.. code-block:: bash

   python -m http.server 8080 --directory docs/_build/html

Then navigate to http://localhost:8080.

Version Management
------------------

Artifacta uses ``bump-my-version`` to manage version numbers across the codebase.

**Show current version:**

.. code-block:: bash

   source venv/bin/activate
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
