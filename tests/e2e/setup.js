import { spawn, execSync } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '../..');

/**
 * Global setup for Playwright E2E tests
 *
 * - Starts the UI server
 * - Cleans/creates fresh database
 * - Runs example script to populate test data
 * - Waits for server to be ready
 */

let serverProcess;

export default async function globalSetup() {
  console.log('[E2E Setup] Starting global test setup...');

  // 1. Clean database for fresh state
  const dbPath = path.join(PROJECT_ROOT, 'data', 'runs.db');
  try {
    await fs.unlink(dbPath);
    console.log('[E2E Setup] ✓ Cleaned database');
  } catch (err) {
    // Database might not exist yet - that's fine
    console.log('[E2E Setup] ✓ No existing database to clean');
  }

  // 2. Start the UI server
  console.log('[E2E Setup] Starting UI server...');
  const baseURL = process.env.ARTIFACTA_URL || 'http://localhost:8000';
  const port = new URL(baseURL).port || '8000';

  serverProcess = spawn(
    'artifacta',
    ['ui', '--port', port],
    {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      shell: true,
    }
  );

  // Handle server output
  serverProcess.stdout.on('data', (data) => {
    if (process.env.DEBUG) {
      console.log(`[Server] ${data}`);
    }
  });

  serverProcess.stderr.on('data', (data) => {
    if (process.env.DEBUG) {
      console.error(`[Server] ${data}`);
    }
  });

  // 3. Wait for server to be ready
  console.log('[E2E Setup] Waiting for server to be ready...');
  await waitForServer(baseURL, 30000); // 30 second timeout
  console.log('[E2E Setup] ✓ Server is ready');

  // Wait a bit longer for database initialization to complete
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // 4. Run example script to populate test data
  console.log('[E2E Setup] Running example script to create test data...');
  try {
    const exampleScript = path.join(PROJECT_ROOT, 'examples', 'core', '02_all_primitives.py');

    const output = execSync(`python ${exampleScript}`, {
      cwd: PROJECT_ROOT,
      encoding: 'utf-8',
      env: {
        ...process.env,
        ARTIFACTA_API_URL: baseURL,
      },
    });
    console.log('[E2E Setup] Example script output:', output.substring(0, 500));
    console.log('[E2E Setup] ✓ Test data created');
  } catch (err) {
    console.error('[E2E Setup] ✗ Failed to create test data:', err.message);
    throw err;
  }

  console.log('[E2E Setup] ✓ Global setup complete\n');

  // Store server process for global teardown
  global.__SERVER_PROCESS__ = serverProcess;
}


/**
 * Wait for server to respond to health check
 */
async function waitForServer(baseURL, timeoutMs) {
  const startTime = Date.now();
  const healthURL = `${baseURL}/health`;

  while (Date.now() - startTime < timeoutMs) {
    try {
      const response = await fetch(healthURL);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'healthy') {
          return;
        }
      }
    } catch (err) {
      // Server not ready yet, continue waiting
    }

    // Wait 500ms before next attempt
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error(`Server did not start within ${timeoutMs}ms`);
}
