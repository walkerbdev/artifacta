import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E Test Configuration
 *
 * Reads base URL from ARTIFACTA_URL environment variable (default: http://localhost:8000)
 *
 * Usage:
 *   npm run test:e2e
 *   ARTIFACTA_URL=http://localhost:8001 npm run test:e2e
 */
export default defineConfig({
  testDir: '../tests/e2e',

  // Global setup and teardown
  globalSetup: '../tests/e2e/setup.js',
  globalTeardown: '../tests/e2e/teardown.js',

  // Test execution settings
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,

  // Reporter
  reporter: 'html',

  // Shared settings for all tests
  use: {
    // Base URL from env var or default
    baseURL: process.env.ARTIFACTA_URL || 'http://localhost:8000',

    // Browser settings
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',

    // Timeouts
    actionTimeout: 10000,
  },

  // Test timeout
  timeout: 30000,

  // Configure projects for different browsers (chromium only for now - fast and headless)
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Run local dev server before starting the tests
  // (disabled - we handle server startup in global setup)
  // webServer: {
  //   command: 'venv/bin/artifacta ui',
  //   url: 'http://localhost:8000',
  //   reuseExistingServer: !process.env.CI,
  // },
});
