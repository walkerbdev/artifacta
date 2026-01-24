/**
 * Global teardown for Playwright E2E tests
 * Stops the server process started in setup.js
 */

export default async function globalTeardown() {
  console.log('\n[E2E Teardown] Stopping server...');

  if (global.__SERVER_PROCESS__) {
    global.__SERVER_PROCESS__.kill('SIGTERM');

    // Wait a moment for graceful shutdown
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Force kill if still running
    try {
      global.__SERVER_PROCESS__.kill('SIGKILL');
    } catch (err) {
      // Process already dead
    }
  }

  console.log('[E2E Teardown] ✓ Teardown complete');
}
